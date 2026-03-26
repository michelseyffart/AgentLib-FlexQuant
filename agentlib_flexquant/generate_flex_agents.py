"""Generate agents for flexibility quantification.

This module provides the FlexAgentGenerator class that creates and configures
flexibility agents.
The agents created include the baseline, positive and negative flexibility agents,
the flexibility indicator and market agents. The agents are created based on the
flex config and the MPC config.
"""
import ast
import atexit
import inspect
import logging
import os

import astor
import black
import json
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Union
from pydantic import FilePath
from agentlib.core.agent import AgentConfig
from agentlib.core.datamodels import AgentVariable
from agentlib.core.errors import ConfigurationError
from agentlib.core.module import BaseModuleConfig
from agentlib.utils import custom_injection, load_config
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib_mpc.models.casadi_model import CasadiModelConfig
from agentlib_mpc.modules.mpc.mpc_full import MPCConfig

from agentlib_mpc.optimization_backends.casadi_.basic import DirectCollocation
from agentlib_mpc.data_structures.casadi_utils import CasadiDiscretizationOptions
import agentlib_flexquant.data_structures.globals as glbs
import agentlib_flexquant.utils.config_management as cmng
from agentlib_flexquant.utils.parsing import SetupSystemModifier
from agentlib_flexquant.data_structures.flexquant import (
    FlexibilityIndicatorConfig,
    FlexibilityMarketConfig,
    FlexQuantConfig,
)
from agentlib_flexquant.data_structures.mpcs import BaselineMPCData, BaseMPCData
from agentlib_flexquant.modules.flexibility_indicator import (
    FlexibilityIndicatorModuleConfig,
)
from agentlib_flexquant.modules.flexibility_market import FlexibilityMarketModuleConfig


class FlexAgentGenerator:
    """Class for generating the flex agents

    orig_mpc_module_config: the config for the original mpc,
                            which has nothing to do with the flexibility quantification
    baseline_mpc_module_config: the config for the baseline mpc
                                for flexibility quantification
    pos_flex_mpc_module_config: the config for the positive flexibility mpc
                                for flexibility quantification
    neg_flex_mpc_module_config: the config for the negative flexibility mpc
                                for flexibility quantification
    indicator_module_config: the config for the indicator for flexibility quantification
    market_module_config: the config for the market for flexibility quantification

    """

    orig_mpc_module_config: MPCConfig
    baseline_mpc_module_config: MPCConfig
    pos_flex_mpc_module_config: MPCConfig
    neg_flex_mpc_module_config: MPCConfig
    indicator_module_config: FlexibilityIndicatorModuleConfig
    market_module_config: FlexibilityMarketModuleConfig

    def __init__(
            self,
            flex_config: Union[str, FilePath, FlexQuantConfig],
            mpc_agent_config: Union[str, FilePath, AgentConfig],
    ):
        self.logger = logging.getLogger(__name__)

        if isinstance(flex_config, str or FilePath):
            self.flex_config_file_name = os.path.basename(flex_config)
        else:
            # provide default name for json
            self.flex_config_file_name = "flex_config.json"
        # load configs
        self.flex_config = load_config.load_config(flex_config,
                                                   config_type=FlexQuantConfig)

        # original mpc agent
        self.orig_mpc_agent_config = load_config.load_config(
            mpc_agent_config, config_type=AgentConfig
        )
        # baseline agent
        self.baseline_mpc_agent_config = self.orig_mpc_agent_config.__deepcopy__()
        self.baseline_mpc_agent_config.id = (self.flex_config.
                                             baseline_config_generator_data.agent_id)
        # pos agent
        self.pos_flex_mpc_agent_config = self.orig_mpc_agent_config.__deepcopy__()
        self.pos_flex_mpc_agent_config.id = (self.flex_config.
                                             shadow_mpc_config_generator_data.
                                             pos_flex.agent_id)
        # neg agent
        self.neg_flex_mpc_agent_config = self.orig_mpc_agent_config.__deepcopy__()
        self.neg_flex_mpc_agent_config.id = (self.flex_config.
                                             shadow_mpc_config_generator_data.
                                             neg_flex.agent_id)

        # original mpc module
        self.orig_mpc_module_config = cmng.get_module(
            config=self.orig_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # baseline module
        self.baseline_mpc_module_config = cmng.get_module(
            config=self.baseline_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # convert agentlib_mpc’s ModuleConfig to flexquant’s ModuleConfig to include additional
        # fields not present in the original
        self.baseline_mpc_module_config = cmng.get_flex_mpc_module_config(
            agent_config=self.baseline_mpc_agent_config,
            module_config=self.baseline_mpc_module_config,
            module_type=self.flex_config.baseline_config_generator_data.module_types[
                self.baseline_mpc_module_config.type
            ]
        )
        # pos module
        self.pos_flex_mpc_module_config = cmng.get_module(
            config=self.pos_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # neg module
        self.neg_flex_mpc_module_config = cmng.get_module(
            config=self.neg_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
        )
        # load indicator config
        self.indicator_config = load_config.load_config(
            self.flex_config.indicator_config, config_type=FlexibilityIndicatorConfig
        )
        # load indicator module config
        self.indicator_agent_config = load_config.load_config(
            self.indicator_config.agent_config, config_type=AgentConfig
        )
        self.indicator_module_config = cmng.get_module(
            config=self.indicator_agent_config, module_type=cmng.INDICATOR_CONFIG_TYPE
        )
        # load market config
        if self.flex_config.market_config:
            self.market_config = load_config.load_config(
                self.flex_config.market_config, config_type=FlexibilityMarketConfig
            )
            # load market module config
            self.market_agent_config = load_config.load_config(
                self.market_config.agent_config, config_type=AgentConfig
            )
            self.market_module_config = cmng.get_module(
                config=self.market_agent_config, module_type=cmng.MARKET_CONFIG_TYPE
            )
        else:
            self.flex_config.market_time = 0

        self.run_config_validations()

    def generate_flex_agents(self) -> list[str]:
        """Generate the configs and the python module for the flexibility agents.

        Returns:
            list of the full path for baseline mpc, pos_flex mpc, neg_flex mpc, indicator
            and market config

        """
        # adapt modules to include necessary communication variables
        baseline_mpc_config = self.adapt_mpc_module_config(
            module_config=self.baseline_mpc_module_config,
            mpc_dataclass=self.flex_config.baseline_config_generator_data,
            agent_id=self.flex_config.baseline_config_generator_data.agent_id,
        )
        pf_mpc_config = self.adapt_mpc_module_config(
            module_config=self.pos_flex_mpc_module_config,
            mpc_dataclass=self.flex_config.shadow_mpc_config_generator_data.pos_flex,
            agent_id=self.flex_config.shadow_mpc_config_generator_data.pos_flex.agent_id,
        )
        nf_mpc_config = self.adapt_mpc_module_config(
            module_config=self.neg_flex_mpc_module_config,
            mpc_dataclass=self.flex_config.shadow_mpc_config_generator_data.neg_flex,
            agent_id=self.flex_config.shadow_mpc_config_generator_data.neg_flex.agent_id,
        )
        indicator_module_config = self.adapt_indicator_module_config(
            module_config=self.indicator_module_config
        )
        if self.flex_config.market_config:
            market_module_config = self.adapt_market_module_config(
                module_config=self.market_module_config
            )

        # dump jsons of the agents including the adapted module configs
        self.append_module_and_dump_agent(
            module=baseline_mpc_config,
            agent=self.baseline_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
            config_name=self.flex_config.baseline_config_generator_data.
            name_of_created_file,
        )
        self.append_module_and_dump_agent(
            module=pf_mpc_config,
            agent=self.pos_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
            config_name=self.flex_config.shadow_mpc_config_generator_data.
            pos_flex.name_of_created_file,
        )
        self.append_module_and_dump_agent(
            module=nf_mpc_config,
            agent=self.neg_flex_mpc_agent_config,
            module_type=cmng.get_orig_module_type(self.orig_mpc_agent_config),
            config_name=self.flex_config.shadow_mpc_config_generator_data.
            neg_flex.name_of_created_file,
        )
        self.append_module_and_dump_agent(
            module=indicator_module_config,
            agent=self.indicator_agent_config,
            module_type=cmng.INDICATOR_CONFIG_TYPE,
            config_name=self.indicator_config.name_of_created_file,
        )
        if self.flex_config.market_config:
            self.append_module_and_dump_agent(
                module=market_module_config,
                agent=self.market_agent_config,
                module_type=cmng.MARKET_CONFIG_TYPE,
                config_name=self.market_config.name_of_created_file,
            )
        # generate python files for the shadow mpcs
        self._generate_flex_model_definition()

        # add new paths to flex config and dump it
        self.adapt_and_dump_flex_config()

        # register the exit function if the corresponding flag is set
        if self.flex_config.delete_files:
            atexit.register(lambda: self._delete_created_files())

        return self.get_config_file_paths()

    def append_module_and_dump_agent(
            self,
            module: BaseModuleConfig,
            agent: AgentConfig,
            module_type: str,
            config_name: str,
    ):
        """Append the given module config to the given agent config and
        dumps the agent config to a json file.

        The json file is named based on the config_name.

        Args:
            module: The module config to be appended.
            agent: The agent config to be updated.
            module_type: The type of the module
            config_name: The name of the json file for module config (e.g. baseline.json)

        """
        # get the module as a dict without default values
        module_dict = cmng.to_dict_and_remove_unnecessary_fields(module=module)
        # write given module to agent config
        for i, agent_module in enumerate(agent.modules):
            if cmng.MODULE_TYPE_DICT[module_type] is cmng.MODULE_TYPE_DICT[
                agent_module["type"]]:
                agent.modules[i] = module_dict

        # dump agent config
        if agent.modules:
            if self.flex_config.overwrite_files:
                try:
                    Path(os.path.join(self.flex_config.flex_files_directory,
                                      config_name)).unlink()
                except OSError:
                    pass
            with open(
                    os.path.join(self.flex_config.flex_files_directory, config_name),
                    "w+",
                    encoding="utf-8",
            ) as f:
                module_json = agent.model_dump_json(exclude_defaults=True)
                f.write(module_json)
        else:
            logging.error("Provided agent config does not contain any modules.")

    def get_config_file_paths(self) -> list[str]:
        """Return a list of paths with the created config files."""
        paths = [
            os.path.join(
                self.flex_config.flex_files_directory,
                self.flex_config.baseline_config_generator_data.
                name_of_created_file,
            ),
            os.path.join(
                self.flex_config.flex_files_directory,
                self.flex_config.shadow_mpc_config_generator_data.pos_flex.
                name_of_created_file,
            ),
            os.path.join(
                self.flex_config.flex_files_directory,
                self.flex_config.shadow_mpc_config_generator_data.neg_flex.
                name_of_created_file,
            ),
            os.path.join(
                self.flex_config.flex_files_directory,
                self.indicator_config.name_of_created_file,
            ),
        ]
        if self.flex_config.market_config:
            paths.append(
                os.path.join(
                    self.flex_config.flex_files_directory,
                    self.market_config.name_of_created_file,
                )
            )
        return paths

    def _delete_created_files(self):
        """Function to run at exit if the files are to be deleted."""
        to_be_deleted = self.get_config_file_paths()
        to_be_deleted.append(
            os.path.join(
                self.flex_config.flex_files_directory,
                self.flex_config_file_name,
            )
        )
        # delete files
        for file in to_be_deleted:
            Path(file).unlink()
        # also delete folder
        Path(self.flex_config.flex_files_directory).rmdir()

    def adapt_mpc_module_config(
            self, module_config: MPCConfig, mpc_dataclass: BaseMPCData, agent_id: str
    ) -> MPCConfig:
        """Adapt the mpc module config for automated flexibility quantification.

        Things adapted among others are:
        - the file name/path of the mpc config file
        - names of the control variables for the shadow mpcs
        - reduce communicated variables of shadow mpcs to outputs
        - add the power variable to the outputs
        - add parameters for the activation and quantification of flexibility

        Args:
            module_config: The module config to be adapted
            mpc_dataclass: The dataclass corresponding to the type of the MPC module.
                           It contains all the extra data necessary for flexibility
                           quantification, which will be used to update the
                           module_config.
            agent_id: agent_id for creating the FlexQuant mpc module config

        Returns:
            The adapted module config

        """
        # allow the module config to be changed
        module_config.model_config["frozen"] = False

        # set new MPC type
        module_config.type = mpc_dataclass.module_types[
            cmng.get_orig_module_type(self.orig_mpc_agent_config)
        ]

        # set the MPC config type from the MPCConfig in agentlib_mpc to the
        # corresponding one in flexquant and add additional fields
        module_config_flex_dict = module_config.model_dump()
        module_config_flex_dict["casadi_sim_time_step"] = (
            self.flex_config.casadi_sim_time_step)
        module_config_flex_dict["power_variable_name"] = (
            self.flex_config.baseline_config_generator_data.power_variable)
        module_config_flex_dict["storage_variable_name"] = (
            self.indicator_module_config.correct_costs.stored_energy_variable)
        module_config_flex = cmng.MODULE_TYPE_DICT[module_config.type](
            **module_config_flex_dict, _agent_id=agent_id
        )

        # HOTFIX due to AgentLib-MPC bug. Needs to be adapted after Objectives
        # in AgentLib-MPC are fixed.
        if module_config_flex.r_del_u is None:
            module_config_flex = module_config_flex.model_copy(update={"r_del_u": {}})

        # allow the module config to be changed
        module_config_flex.model_config["frozen"] = False

        module_config_flex.module_id = mpc_dataclass.module_id

        # set new id (needed for plotting)
        module_config_flex.module_id = mpc_dataclass.module_id
        # update optimization backend to use the created mpc files and classes
        module_config_flex.optimization_backend["model"]["type"] = {
            "file": os.path.join(
                self.flex_config.flex_files_directory,
                mpc_dataclass.created_flex_mpcs_file,
            ),
            "class_name": mpc_dataclass.class_name,
        }
        # extract filename from results file and update it with
        # suffix and parent directory
        result_filename = Path(
            module_config_flex.optimization_backend["results_file"]
        ).name.replace(".csv", mpc_dataclass.results_suffix)
        full_path = self.flex_config.results_directory / result_filename
        module_config_flex.optimization_backend["results_file"] = str(full_path)
        # change cia backend to custom backend of flexquant
        if module_config_flex.optimization_backend["type"] == "casadi_cia":
            module_config_flex.optimization_backend["type"] = \
                "agentlib_flexquant.casadi_cia_cons"
        if (module_config_flex.optimization_backend["type"] ==
                "agentlib_flexquant.casadi_cia_cons"):
            module_config_flex.optimization_backend["market_time"] = (
                self.flex_config.market_time)

        # add the full control trajectory output from the baseline as input for the
        # shadow mpcs, they are directly included in the optimization problem
        if not isinstance(mpc_dataclass, BaselineMPCData):
            for control in module_config_flex.controls:
                module_config_flex.inputs.append(
                    MPCVariable(
                        name=control.name + glbs.full_trajectory_suffix,
                        value=None,
                        type="pd.Series",
                    )
                )
                # add full control names to shadow MPC config for inputs tracking
                module_config_flex.full_control_names.append(
                    control.name + glbs.full_trajectory_suffix)
                # change the alias of control variable in shadow mpc to
                # prevent it from triggering the wrong callback
                control.alias = control.name + glbs.shadow_suffix
            # also include binary controls
            if hasattr(module_config_flex, "binary_controls"):
                for control in module_config_flex.binary_controls:
                    module_config_flex.inputs.append(
                        MPCVariable(
                            name=control.name + glbs.full_trajectory_suffix,
                            value=None,
                            type="pd.Series",
                        )
                    )
                    # add full control names to shadow MPC config for inputs tracking
                    module_config_flex.full_control_names.append(
                        control.name + glbs.full_trajectory_suffix)
                    # change the alias of control variable in shadow mpc to
                    # prevent it from triggering the wrong callback
                    control.alias = control.name + glbs.shadow_suffix
            # only communicate outputs for the shadow mpcs
            module_config_flex.shared_variable_fields = ["outputs"]

            # In addition to creating the full control variables, the inputs
            # and states  of the Baseline are communicated to the Shadow MPC
            # to ensure synchronisation. Therefore, all inputs and states of
            # the Baseline are added to the Shadow MPCs with an alias
            baseline_names = {inp.name for inp in
                              self.baseline_mpc_module_config.inputs}
            for i, input in enumerate(module_config_flex.inputs):
                if input.name in baseline_names:
                    module_config_flex.inputs[i].alias = (
                            input.name + glbs.base_vars_to_communicate_suffix)

            # add Baseline input names to shadow MPC config for inputs tracking
            # if Baseline variable is also set in config_inputs_appendix this is
            # due to overwriting the alias, so variable should not be added here
            appendix_names = {inp.name for inp in mpc_dataclass.config_inputs_appendix}
            module_config_flex.baseline_input_names = [
                input.name + glbs.base_vars_to_communicate_suffix for input in
                self.baseline_mpc_module_config.inputs
                if input.name not in appendix_names]

            # add custom input names for the shadow MPC to track. Here, the
            # communication suffix is not added, as  the user is free to define
            # custom inputs as desired.
            # Exclude in_provision, as this is not regularly set and would prevent
            # the do_step of the shadow MPC.
            module_config_flex.custom_input_names.extend([
                {"name": input.name, "alias": input.alias}
                for input in mpc_dataclass.config_inputs_appendix
                if input.name not in [glbs.PROVISION_VAR_NAME]
            ])

            for i, state in enumerate(module_config_flex.states):
                if state in self.baseline_mpc_module_config.states:
                    module_config_flex.states[i].alias = (
                            state.name + glbs.base_vars_to_communicate_suffix)
            # add Baseline state names to shadow MPC config for inputs tracking
            module_config_flex.baseline_state_names = [
                state.name + glbs.base_vars_to_communicate_suffix for state in
                self.baseline_mpc_module_config.states]
            module_config_flex.baseline_agent_id = (
                self.flex_config.baseline_config_generator_data.agent_id)

        else:
            # all the variables here are added to the custom MPCConfig of
            # FlexQuant to avoid them being added to the optimization problem
            # add full_controls trajectory as AgentVariable to the config of
            # Baseline mpc
            for control in module_config_flex.controls:
                module_config_flex.full_controls.append(
                    AgentVariable(
                        name=control.name + glbs.full_trajectory_suffix,
                        alias=control.name + glbs.full_trajectory_suffix,
                        shared=True,
                    )
                )
            if hasattr(module_config_flex, "binary_controls"):
                for binary_controls in module_config_flex.binary_controls:
                    module_config_flex.full_controls.append(
                        AgentVariable(
                            name=binary_controls.name + glbs.full_trajectory_suffix,
                            alias=binary_controls.name + glbs.full_trajectory_suffix,
                            shared=True,
                        )
                    )
                # add full controls to custom cia backend to constrain
                # during market time
                if (module_config_flex.optimization_backend["type"] ==
                        "agentlib_flexquant.casadi_cia_cons"):
                    module_config_flex.optimization_backend["full_controls_dict"] = (
                        dict(zip([var.name for var in module_config_flex.full_controls],
                                 [None] * len(module_config_flex.full_controls))
                             ))
            # add input and states copy variables which send the Baseline inputs
            # to the shadow MPC
            for input in module_config_flex.inputs:
                module_config_flex.vars_to_communicate.append(
                    AgentVariable(
                        name=input.name + glbs.base_vars_to_communicate_suffix,
                        alias=input.name + glbs.base_vars_to_communicate_suffix,
                        shared=True,
                    )
                )
            for state in module_config_flex.states:
                module_config_flex.vars_to_communicate.append(
                    AgentVariable(
                        name=state.name + glbs.base_vars_to_communicate_suffix,
                        alias=state.name + glbs.base_vars_to_communicate_suffix,
                        shared=True,
                    )
                )

        module_config_flex.set_outputs = True
        # add outputs for the power variables, for easier handling create a lookup dict
        output_dict = {output.name: output for output in module_config_flex.outputs}
        if self.flex_config.baseline_config_generator_data.power_variable in output_dict:
            output_dict[
                self.flex_config.baseline_config_generator_data.power_variable
            ].alias = mpc_dataclass.power_alias
        else:
            module_config_flex.outputs.append(
                MPCVariable(
                    name=self.flex_config.baseline_config_generator_data.power_variable,
                    alias=mpc_dataclass.power_alias,
                )
            )
        # add or change alias for stored energy variable
        if self.indicator_module_config.correct_costs.enable_energy_costs_correction:
            output_dict[
                self.indicator_module_config.correct_costs.stored_energy_variable
            ].alias = mpc_dataclass.stored_energy_alias

        # add extra inputs needed for activation of flex or custom cost functions
        existing_input_names = {inp.name: idx for idx, inp in
                                enumerate(module_config_flex.inputs)}
        for appendix_inp in mpc_dataclass.config_inputs_appendix.copy():
            # If variable already exists in the config
            if appendix_inp.name in existing_input_names:
                self.logger.warning(f"The given variable {appendix_inp.name} in the "
                                    f"config_inputs_appendix already exists in the MPC "
                                    f"model. I am updating the alias of the existing "
                                    f"variable to {appendix_inp.alias} (provided by you). "
                                    f"However, this can still cause issues down the line "
                                    f"if the alias is not chosen wisely.")
                # Update only the alias of the existing input
                idx = existing_input_names[appendix_inp.name]
                existing_inp = module_config_flex.inputs[idx]
                inp_dict = existing_inp.dict()
                inp_dict["alias"] = appendix_inp.alias
                module_config_flex.inputs[idx] = type(existing_inp)(**inp_dict)
                # Remove variable from appendix list to avoid creation during parsing
                mpc_dataclass.config_inputs_appendix.remove(appendix_inp)
            else:
                # Add the new input
                module_config_flex.inputs.append(appendix_inp)

        # add extra parameters needed for activation of flex or custom weights
        for var in mpc_dataclass.config_parameters_appendix:
            if var.name in self.flex_config.model_fields:
                var.value = getattr(self.flex_config, var.name)
            if var.name in self.flex_config.baseline_config_generator_data.model_fields:
                var.value = getattr(self.flex_config.baseline_config_generator_data,
                                    var.name)
        module_config_flex.parameters.extend(mpc_dataclass.config_parameters_appendix)

        # freeze the config again
        module_config_flex.model_config["frozen"] = True

        return module_config_flex

    def adapt_indicator_module_config(
            self, module_config: FlexibilityIndicatorModuleConfig
    ) -> FlexibilityIndicatorModuleConfig:
        """Adapt the indicator module config for automated flexibility
        quantification.

        """
        # append user-defined price var to indicator module config
        module_config.inputs.append(
            AgentVariable(
                name=module_config.price_variable,
                unit="ct/kWh",
                type="pd.Series",
                description="electricity price",
            )
        )
        module_config.inputs.append(
            AgentVariable(
                name=module_config.price_variable_feed_in,
                unit="ct/kWh",
                type="pd.Series",
                description="electricity feed-in price",
            )
        )
        # allow the module config to be changed
        module_config.model_config["frozen"] = False
        for parameter in module_config.parameters:
            if parameter.name == glbs.PREP_TIME:
                parameter.value = self.flex_config.prep_time
            if parameter.name == glbs.MARKET_TIME:
                parameter.value = self.flex_config.market_time
            if parameter.name == glbs.FLEX_EVENT_DURATION:
                parameter.value = self.flex_config.flex_event_duration
            if parameter.name == glbs.TIME_STEP:
                parameter.value = self.baseline_mpc_module_config.time_step
            if parameter.name == glbs.PREDICTION_HORIZON:
                parameter.value = self.baseline_mpc_module_config.prediction_horizon
            if parameter.name == glbs.COLLOCATION_TIME_GRID:
                dis_op = self.baseline_mpc_module_config.optimization_backend[
                    "discretization_options"
                ]
                parameter.value = self.get_collocation_time_grid(
                    discretization_options=dis_op
                )
        # set power unit
        module_config.power_unit = (
            self.flex_config.baseline_config_generator_data.power_unit)
        module_config.results_file = (
                self.flex_config.results_directory / module_config.results_file.name
        )
        module_config.model_config["frozen"] = True
        return module_config

    def adapt_market_module_config(
            self, module_config: FlexibilityMarketModuleConfig
    ) -> FlexibilityMarketModuleConfig:
        """Adapt the market module config for automated flexibility quantification."""
        # allow the module config to be changed
        module_config.model_config["frozen"] = False
        for field in module_config.__fields__:
            if field in self.market_module_config.__fields__.keys():
                module_config.__setattr__(field, getattr(self.market_module_config,
                                                         field))
        module_config.results_file = (
                self.flex_config.results_directory / module_config.results_file.name
        )
        for parameter in module_config.parameters:
            if parameter.name == glbs.COLLOCATION_TIME_GRID:
                dis_op = self.baseline_mpc_module_config.optimization_backend[
                    "discretization_options"
                ]
                parameter.value = self.get_collocation_time_grid(
                    discretization_options=dis_op
                )
            if parameter.name == glbs.TIME_STEP:
                parameter.value = self.baseline_mpc_module_config.time_step
        module_config.model_config["frozen"] = True
        return module_config

    def adapt_and_dump_flex_config(self):
        """Update flex_config to reference the newly generated market/indicator agent configs and
        dump the updated flex configuration to disk.

        This method replaces the market and indicator configuration entries in ``self.flex_config``
        with the internally created ``self.market_config`` and ``self.indicator_config``. If a
        market configuration is present, its ``agent_config`` attribute is updated to the path of
        the newly created market agent config file under ``flex_files_directory``. Likewise, the
        indicator configuration's ``agent_config`` attribute is set to the path of the newly
        created indicator agent config file. These paths correspond to the new locations of the
        market or indicator config files when they were originally provided to the
        ``FlexAgentGenerator`` as file paths.

        After updating these paths, the complete ``flex_config`` is serialized (excluding default
        values) and written as JSON to ``flex_files_directory / flex_config_file_name`` so that
        subsequent runs can use the resolved configuration directly.
        """
        # store market and indicator with file path of created agent config
        if self.flex_config.market_config:
            self.flex_config.market_config = self.market_config
            self.flex_config.market_config.agent_config = os.path.join(
                self.flex_config.flex_files_directory,
                self.market_config.name_of_created_file)
        self.flex_config.indicator_config = self.indicator_config
        self.flex_config.indicator_config.agent_config = os.path.join(
            self.flex_config.flex_files_directory,
            self.indicator_config.name_of_created_file)
        # save flex config to created flex files
        with open(os.path.join(self.flex_config.flex_files_directory,
                               self.flex_config_file_name),
                  "w", encoding="utf-8", ) as f:
            config_json = self.flex_config.model_dump_json(exclude_defaults=True)
            f.write(config_json)

    def get_collocation_time_grid(self, discretization_options: dict):
        """Get the mpc output collocation grid over the horizon"""
        # get the mpc time grid configuration
        time_step = self.baseline_mpc_module_config.time_step
        prediction_horizon = self.baseline_mpc_module_config.prediction_horizon
        # get the collocation configuration
        collocation_method = discretization_options["collocation_method"]
        collocation_order = discretization_options["collocation_order"]
        # get the collocation points
        options = CasadiDiscretizationOptions(
            collocation_order=collocation_order, collocation_method=collocation_method
        )
        collocation_points = DirectCollocation(options=
                                               options)._collocation_polynomial().root
        # compute the mpc output collocation grid
        discretization_points = np.arange(0, time_step * prediction_horizon, time_step)
        collocation_time_grid = (
                discretization_points[:, None] + collocation_points * time_step
        ).ravel()
        collocation_time_grid = collocation_time_grid[
            ~np.isin(collocation_time_grid, discretization_points)
        ]
        collocation_time_grid = collocation_time_grid.tolist()
        return collocation_time_grid

    def _generate_flex_model_definition(self):
        """Generate a python module for negative and positive flexibility agents
        from the Baseline MPC model."""
        output_file = os.path.join(
            self.flex_config.flex_files_directory,
            self.flex_config.baseline_config_generator_data.created_flex_mpcs_file,
        )
        opt_backend = self.orig_mpc_module_config.optimization_backend["model"]["type"]

        # Extract the config class of the casadi model to check cost functions
        config_class = inspect.get_annotations(custom_injection(opt_backend))["config"]
        # Get custom module fields provided by the user and add them
        model_fields = self.baseline_mpc_module_config.optimization_backend["model"]
        _ = model_fields.pop("type")
        config_instance = config_class(**model_fields)
        # The " + " is just there to simplify the validation, it does not affect
        # the generated code
        self.check_variables_in_casadi_config(
            config_instance,
            self.flex_config.shadow_mpc_config_generator_data.neg_flex.flex_cost_function +
            (
                " + " + self.flex_config.shadow_mpc_config_generator_data.neg_flex.flex_cost_function_appendix
                if self.flex_config.shadow_mpc_config_generator_data.neg_flex.flex_cost_function_appendix else ""),
            shadow_mpc_type="neg_flex"
        )
        self.check_variables_in_casadi_config(
            config_instance,
            self.flex_config.shadow_mpc_config_generator_data.pos_flex.flex_cost_function +
            (
                " + " + self.flex_config.shadow_mpc_config_generator_data.pos_flex.flex_cost_function_appendix
                if self.flex_config.shadow_mpc_config_generator_data.pos_flex.flex_cost_function_appendix else ""),
            shadow_mpc_type="pos_flex"
        )

        # parse mpc python file
        with open(opt_backend["file"], "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)

        # create modifiers for python file
        modifier_base = SetupSystemModifier(
            mpc_data=self.flex_config.baseline_config_generator_data,
            controls=self.baseline_mpc_module_config.controls,
            binary_controls=self.baseline_mpc_module_config.binary_controls
            if hasattr(self.baseline_mpc_module_config, "binary_controls")
            else None,
        )
        modifier_pos = SetupSystemModifier(
            mpc_data=self.flex_config.shadow_mpc_config_generator_data.pos_flex,
            controls=self.pos_flex_mpc_module_config.controls,
            binary_controls=self.pos_flex_mpc_module_config.binary_controls
            if hasattr(self.pos_flex_mpc_module_config, "binary_controls")
            else None,
        )
        modifier_neg = SetupSystemModifier(
            mpc_data=self.flex_config.shadow_mpc_config_generator_data.neg_flex,
            controls=self.neg_flex_mpc_module_config.controls,
            binary_controls=self.neg_flex_mpc_module_config.binary_controls
            if hasattr(self.neg_flex_mpc_module_config, "binary_controls")
            else None,
        )
        # run the modification
        modified_tree_base = modifier_base.visit(deepcopy(tree))
        modified_tree_pos = modifier_pos.visit(deepcopy(tree))
        modified_tree_neg = modifier_neg.visit(deepcopy(tree))
        # combine modifications to one file
        modified_tree = ast.Module(body=[], type_ignores=[])
        modified_tree.body.extend(
            modified_tree_base.body + modified_tree_pos.body + modified_tree_neg.body
        )
        modified_source = astor.to_source(modified_tree)
        # Use black to format the generated code
        formatted_code = black.format_str(modified_source, mode=black.FileMode())

        if self.flex_config.overwrite_files:
            try:
                Path(
                    os.path.join(
                        self.flex_config.flex_files_directory,
                        self.flex_config.baseline_config_generator_data.created_flex_mpcs_file,
                    )
                ).unlink()
            except OSError:
                pass

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_code)

    def check_variables_in_casadi_config(self, config: CasadiModelConfig, expr: str,
                                         shadow_mpc_type: str):
        """Check if all variables in the expression are defined in the config.

        Args:
            config: casadi model config.
            expr: The expression to check.

        Raises:
            ValueError: If any variable in the expression is not defined in the config.

        """
        variables_in_config = set(config.get_variable_names())
        variables_in_cost_function = set(ast.walk(ast.parse(expr)))
        variables_in_cost_function = {
            node.attr for node in variables_in_cost_function
            if isinstance(node, ast.Attribute)
        }
        flex_config_data = (self.flex_config.shadow_mpc_config_generator_data.pos_flex
                            if shadow_mpc_type == "pos_flex"
                            else self.flex_config.shadow_mpc_config_generator_data.neg_flex)
        variables_newly_created = set(
            [par.name for par in flex_config_data.config_parameters_appendix] +
            [inp.name for inp in flex_config_data.config_inputs_appendix]
        )

        unknown_vars = (variables_in_cost_function - variables_in_config -
                        variables_newly_created)
        if unknown_vars:
            self.logger.warning(f"Unknown variables in new cost function: "
                                f"{unknown_vars}. This might cause problems with "
                                f"the optimization backend.")

    def run_config_validations(self):
        """Function to validate integrity of user-supplied flex config.

        Since the validation depends on interactions between multiple configurations,
        it is performed within this function rather than using Pydantic’s built-in
        validators for individual configurations.

        The following checks are performed:
        1. Ensures the specified power variable exists in the MPC model outputs.
        2. Ensures the specified comfort variable exists in the MPC model states.
        3. Validates that the stored energy variable exists in MPC outputs if
        energy cost correction is enabled.
        4. Verifies the supported collocation method is used; otherwise,
        switches to 'legendre' and raises a warning.
        5. Ensures that the sum of prep time, market time, and flex event duration
        does not exceed the prediction horizon.
        6. Ensures market time equals the MPC model time step if market config is
        present.
        7. Ensures that all flex time values are multiples of the MPC model time step.
        8. Checks for mismatches between time-related parameters in the flex/MPC and
        indicator configs and issues warnings when discrepancies exist, using the
        flex/MPC config values as the source of truth.

        """
        # check if the power variable exists in the mpc config
        power_var = self.flex_config.baseline_config_generator_data.power_variable
        if power_var not in [output.name for output in
                             self.baseline_mpc_module_config.outputs]:
            raise ConfigurationError(
                f"Given power variable {power_var} is not defined "
                f"as output in baseline mpc config."
            )

        # check if the comfort variable exists in the mpc slack variables
        mod_type = self.baseline_mpc_module_config.optimization_backend["model"]["type"]
        if self.flex_config.baseline_config_generator_data.comfort_variable:
            file_path = mod_type["file"]
            class_name = mod_type["class_name"]
            # Get the class
            dynamic_class = cmng.get_class_from_file(file_path, class_name)
            if self.flex_config.baseline_config_generator_data.comfort_variable not in [
                state.name for state in dynamic_class().states
            ]:
                raise ConfigurationError(
                    f"Given comfort variable "
                    f"{self.flex_config.baseline_config_generator_data.comfort_variable} "
                    f"is not defined as state in baseline mpc config."
                )

        # check if the energy storage variable exists in the mpc config
        if self.indicator_module_config.correct_costs.enable_energy_costs_correction:
            if self.indicator_module_config.correct_costs.stored_energy_variable not in [
                output.name for output in self.baseline_mpc_module_config.outputs
            ]:
                raise ConfigurationError(
                    f"The stored energy variable "
                    f"{self.indicator_module_config.correct_costs.stored_energy_variable} "
                    f"is not defined in baseline mpc config. "
                    f"It must be defined in the base MPC model and config as output "
                    f"if the correction of costs is enabled."
                )

        # raise warning if unsupported collocation method is used and change
        # to supported method
        if (
                "collocation_method"
                not in self.baseline_mpc_module_config.optimization_backend[
            "discretization_options"]
        ):
            raise ConfigurationError(
                "Please use collocation as discretization method and define the "
                "collocation_method in the mpc config"
            )
        else:
            collocation_method = self.baseline_mpc_module_config.optimization_backend[
                "discretization_options"
            ]["collocation_method"]
            if collocation_method != "legendre":
                self.logger.warning(
                    "Collocation method %s is not supported. Switching to "
                    "method legendre.",
                    collocation_method,
                )
                self.baseline_mpc_module_config.optimization_backend[
                    "discretization_options"][
                    "collocation_method"
                ] = "legendre"
                self.pos_flex_mpc_module_config.optimization_backend[
                    "discretization_options"][
                    "collocation_method"
                ] = "legendre"
                self.neg_flex_mpc_module_config.optimization_backend[
                    "discretization_options"][
                    "collocation_method"
                ] = "legendre"

        # time data validations
        flex_times = {
            glbs.PREP_TIME: self.flex_config.prep_time,
            glbs.MARKET_TIME: self.flex_config.market_time,
            glbs.FLEX_EVENT_DURATION: self.flex_config.flex_event_duration,
        }
        mpc_times = {
            glbs.TIME_STEP: self.baseline_mpc_module_config.time_step,
            glbs.PREDICTION_HORIZON: self.baseline_mpc_module_config.prediction_horizon,
        }
        # total time length check (prep+market+flex_event)
        if (sum(flex_times.values()) > mpc_times["time_step"] *
                mpc_times["prediction_horizon"]):
            raise ConfigurationError(
                "Market time + prep time + flex event duration "
                "can not exceed the prediction horizon."
            )
        # market time val check
        if self.flex_config.market_config:
            if flex_times["market_time"] % mpc_times["time_step"] != 0:
                raise ConfigurationError(
                    "Market time must be an integer multiple of the time step."
                )
        # check for divisibility of flex_times by time_step
        for name, value in flex_times.items():
            if value % mpc_times["time_step"] != 0:
                raise ConfigurationError(
                    f"{name} is not a multiple of the time step. Please redefine."
                )
        # raise warning if parameter value in flex indicator module config differs from
        # value in flex config/ baseline mpc module config
        for parameter in self.indicator_module_config.parameters:
            if parameter.value is not None:
                if parameter.name in flex_times:
                    flex_value = flex_times[parameter.name]
                    if parameter.value != flex_value:
                        self.logger.warning(
                            "Value mismatch for %s in flex config (field) "
                            "and indicator module config (parameter). "
                            "Flex config value will be used.",
                            parameter.name,
                        )
                elif parameter.name in mpc_times:
                    mpc_value = mpc_times[parameter.name]
                    if parameter.value != mpc_value:
                        self.logger.warning(
                            "Value mismatch for %s in baseline MPC module "
                            "config (field) and indicator module config (parameter). "
                            "Baseline MPC module config value will be used.",
                            parameter.name,
                        )

    def adapt_sim_results_path(self, simulator_agent_config: Union[str, Path],
                               save_name_suffix: str = "") -> Union[str, Path]:
        """
        Optional helper function to adapt file path for simulator results in sim config,
        so that sim results land in the same results directory as flex results.

        Args:
            simulator_agent_config: Path to the simulator agent config JSON file.
            save_name_suffix: Suffix added to the newly created sim_config file.

        Returns:
            The updated simulator config dictionary with the modified result file path.

        Raises:
            FileNotFoundError: If the specified config file does not exist.

        """
        simulator_agent_config = Path(simulator_agent_config)
        # open config and extract sim module
        with open(simulator_agent_config, "r", encoding="utf-8") as f:
            sim_config = json.load(f)
        sim_module_config = next(
            (module for module in sim_config["modules"] if
             module["type"] == "simulator"), None)
        # convert filename string to path and extract the name
        sim_file_name = Path(sim_module_config["result_filename"]).name
        # set results path so that sim results lands in same directory
        # as flex result CSVs
        sim_module_config["result_filename"] = str(
            self.flex_config.results_directory / sim_file_name
        )
        try:
            with open(Path(str(simulator_agent_config.parent) + "\\" +
                           str(simulator_agent_config.stem) + save_name_suffix + ".json"),
                      "w", encoding="utf-8") as f:
                json.dump(sim_config, f, indent=4)
            return Path(str(simulator_agent_config.parent) + "\\" +
                        str(simulator_agent_config.stem) + save_name_suffix + ".json")
        except Exception as e:
            raise Exception(f"Could not adapt and create a new simulation config "
                            f"due to: {e}. "
                            f"Please check {simulator_agent_config} and "
                            f"'{save_name_suffix}'")
