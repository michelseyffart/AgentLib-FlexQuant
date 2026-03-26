import importlib.util
import inspect
import math
import os
from abc import ABCMeta
from copy import deepcopy
from typing import TypeVar

from agentlib.core.agent import AgentConfig
from agentlib.core.module import BaseModuleConfig
from agentlib.modules import get_all_module_types

T = TypeVar("T", bound=BaseModuleConfig)

all_module_types = get_all_module_types(["agentlib_mpc", "agentlib_flexquant"])
# remove ML models, since import takes ages
all_module_types.pop("agentlib_mpc.ann_trainer")
all_module_types.pop("agentlib_mpc.gpr_trainer")
all_module_types.pop("agentlib_mpc.linreg_trainer")
all_module_types.pop("agentlib_mpc.ml_simulator")
all_module_types.pop("agentlib_mpc.set_point_generator")
# remove clone since not used
all_module_types.pop("clonemap")

# dictionary mapping the module name to the module config (ModelMetaclass)
MODULE_TYPE_DICT = {
    name: inspect.get_annotations(class_type.import_class())["config"]
    for name, class_type in all_module_types.items()
}
# dictionary mapping the module name to the module (ModuleImport)
MODULE_NAME_DICT = all_module_types

MPC_CONFIG_TYPE: str = "agentlib_mpc.mpc"
BASELINEMPC_CONFIG_TYPE: str = "agentlib_flexquant.baseline_mpc"
SHADOWMPC_CONFIG_TYPE: str = "agentlib_flexquant.shadow_mpc"
BASELINEMINLPMPC_CONFIG_TYPE: str = "agentlib_flexquant.baseline_minlp_mpc"
SHADOWMINLPMPC_CONFIG_TYPE: str = "agentlib_flexquant.shadow_minlp_mpc"
INDICATOR_CONFIG_TYPE: str = "agentlib_flexquant.flexibility_indicator"
MARKET_CONFIG_TYPE: str = "agentlib_flexquant.flexibility_market"
SIMULATOR_CONFIG_TYPE: str = "simulator"


def get_module_type_matching_dict(dictionary: dict) -> (dict, dict):
    """Create two dictionaries, which map the modules types of the agentlib_mpc modules
    to those of the flexquant modules.

    This is done by using the MODULE_NAME_DICT.

    """
    # Create dictionaries to store keys grouped by values
    value_to_keys = {}
    for k, v in dictionary.items():
        if k.startswith("agentlib_mpc."):
            if v not in value_to_keys:
                value_to_keys[v] = {"agentlib": [], "flex": []}
            value_to_keys[v]["agentlib"].append(k)
        if k.startswith("agentlib_flexquant."):
            # find the parent class of the module in the flexquant in agentlib_mpc
            for vv in value_to_keys:
                if vv.import_class() is v.import_class().__bases__[0]:
                    value_to_keys[vv]["flex"].append(k)
                    break

    # Create result dictionaries
    baseline_matches = {}
    shadow_matches = {}

    for v, keys in value_to_keys.items():
        # Check if we have both agentlib and flexibility keys for this value
        if keys["agentlib"] and keys["flex"]:
            # Map each agentlib key to corresponding flexibility key
            for agent_key in keys["agentlib"]:
                for flex_key in keys["flex"]:
                    if "baseline" in flex_key:
                        baseline_matches[agent_key] = flex_key
                    elif "shadow" in flex_key:
                        shadow_matches[agent_key] = flex_key

    return baseline_matches, shadow_matches


BASELINE_MODULE_TYPE_DICT, SHADOW_MODULE_TYPE_DICT = (
    get_module_type_matching_dict(MODULE_NAME_DICT))


def get_orig_module_type(config: AgentConfig) -> str:
    """Return the config type of the original MPC."""
    for module in config.modules:
        if module["type"].startswith("agentlib_mpc"):
            return module["type"]


def get_module(config: AgentConfig, module_type: str) -> T:
    """Extracts a module from a config based on its name."""
    for module in config.modules:
        if module["type"] == module_type:
            # deepcopy -> avoid changing the original config, when editing the module
            # deepcopy the args of the constructor instead of the module object,
            # because the simulator module exceeds the recursion limit
            config_id = deepcopy(config.id)
            mod = deepcopy(module)
            return MODULE_TYPE_DICT[mod["type"]](**mod, _agent_id=config_id)
    else:
        raise ModuleNotFoundError(
            f"Module type {module['type']} not found in " f"agentlib and its plug ins."
        )


def get_flex_mpc_module_config(
    agent_config: AgentConfig, module_config: BaseModuleConfig, module_type: str
):
    """Get a flexquant module config from an original agentlib module config."""
    config_dict = module_config.model_dump()
    config_dict["type"] = module_type
    flex_config_dict = MODULE_TYPE_DICT[module_type](**config_dict,
                                                     _agent_id=agent_config.id)
    # HOTFIX due to AgentLib-MPC bug. Needs to be adapted after Objectives
    # in AgentLib-MPC are fixed.
    if flex_config_dict.r_del_u is None:
        flex_config_dict = flex_config_dict.model_copy(update={"r_del_u": {}})
    return flex_config_dict


def to_dict_and_remove_unnecessary_fields(module: BaseModuleConfig) -> dict:
    """Remove unnecessary fields from the module to keep the created json simple."""
    excluded_fields = [
        "rdf_class",
        "source",
        "type",
        "timestamp",
        "description",
        "unit",
        "clip",
        "shared",
        "interpolation_method",
        "allowed_values",
    ]

    def check_bounds(parameter):
        delete_list = excluded_fields.copy()
        if parameter.lb == -math.inf:
            delete_list.append("lb")
        if parameter.ub == math.inf:
            delete_list.append("ub")
        return delete_list

    parent_dict = module.model_dump(exclude_defaults=True)
    # update every variable with a dict excluding the defined fields
    if "parameters" in parent_dict:
        parent_dict["parameters"] = [
            parameter.dict(exclude=check_bounds(parameter)) for
            parameter in module.parameters
        ]
    if "inputs" in parent_dict:
        parent_dict["inputs"] = [input.dict(exclude=check_bounds(input)) for
                                 input in module.inputs]
    if "outputs" in parent_dict:
        parent_dict["outputs"] = [
            output.dict(exclude=check_bounds(output)) for output in module.outputs
        ]
    if "controls" in parent_dict:
        parent_dict["controls"] = [
            control.dict(exclude=check_bounds(control)) for control in module.controls
        ]
    if "binary_controls" in parent_dict:
        parent_dict["binary_controls"] = [
            binary_control.dict(exclude=check_bounds(binary_control))
            for binary_control in module.binary_controls
        ]
    if "states" in parent_dict:
        parent_dict["states"] = [state.dict(exclude=check_bounds(state)) for
                                 state in module.states]
    if "full_controls" in parent_dict:
        parent_dict["full_controls"] = [
            full_control.dict(
                exclude=(lambda ex:
                         ex.remove("shared") or ex)(check_bounds(full_control))
            )
            for full_control in module.full_controls
        ]
    if "vars_to_communicate" in parent_dict:
        parent_dict["vars_to_communicate"] = [
            var_to_communicate.dict(
                exclude=(lambda ex:
                         ex.remove("shared") or ex)(check_bounds(var_to_communicate))
            )
            for var_to_communicate in module.vars_to_communicate
        ]

    return parent_dict


def get_class_from_file(file_path: str, class_name: str) -> ABCMeta:
    # Get the absolute path if needed
    abs_path = os.path.abspath(file_path)

    # Get the module name from the file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module specification
    spec = importlib.util.spec_from_file_location(module_name, abs_path)

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Execute the module
    spec.loader.exec_module(module)

    # Get the class from the module
    target_class = getattr(module, class_name)

    return target_class
