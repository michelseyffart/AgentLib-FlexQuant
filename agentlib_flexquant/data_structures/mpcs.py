"""
Data models for MPC configurations in flexibility quantification.

This module defines Pydantic data models that encapsulate configuration parameters
for baseline, positive flexibility, and negative flexibility MPC controllers used
in flexquant. The models handle file paths, module configurations, variable
mappings, and optimization weights for MPC implementations.
"""
import pydantic
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from pydantic import model_validator, field_serializer, Field

import agentlib_flexquant.data_structures.globals as glbs
import agentlib_flexquant.utils.config_management as cmng

excluded_fields = [
        "rdf_class",
        "source",
        "type",
        "timestamp",
        "description",
        "unit",
        "clip",
        "interpolation_method",
        "allowed_values",
    ]

default_parameters = [
        MPCVariable(
            name=glbs.PREP_TIME,
            value=0,
            unit="s",
            type="int",
            description="Preparation time before switching objective",
        ),
        MPCVariable(
            name=glbs.FLEX_EVENT_DURATION,
            value=0,
            unit="s",
            type="int",
            description="Duration of the flexibility event",
        ),
        MPCVariable(
            name=glbs.MARKET_TIME,
            value=0,
            unit="s",
            type="int",
            description="Market time associated with the objective switch",
        ),
    ]

default_inputs = [
        MPCVariable(
            name=glbs.PROVISION_VAR_NAME,
            value=False,
            type="bool",
            description="Flag indicating whether flexibility should be provisioned",
        ),
    ]

def _ensure_defaults_in_appendix(
    data: dict,
    field_name: str,
    default_variables: list[MPCVariable],
) -> dict:
    """
    Helper to ensure that all default_variables are present in
    data[field_name], based on their ``name`` attribute.
    """
    provided_variables = data.get(field_name, [])
    provided_variables = [MPCVariable(**var) if isinstance(var, dict) else var for var in provided_variables]
    provided_names = {param.name for param in provided_variables}
    for default_var in default_variables:
        if default_var.name not in provided_names:
            provided_variables.append(default_var)

    data[field_name] = provided_variables
    return data
    

class BaseMPCData(pydantic.BaseModel):
    """Base class containing necessary data for the code creation
    of the different mpcs

    """

    # files and paths
    created_flex_mpcs_file: str = "flex_agents.py"
    name_of_created_file: str
    results_suffix: str
    # modules
    module_types: dict
    class_name: str
    module_id: str
    agent_id: str
    # variables
    power_alias: str
    stored_energy_alias: str
    config_inputs_appendix: list[MPCVariable] = Field(
        default=[],
        description="Inputs, which are appended to the MPCs' config "
                    "(.json file and ConfigClass).")
    config_parameters_appendix: list[MPCVariable] = Field(
        default=[],
        description="Parameters, which are appended to the MPCs' config "
                    "(.json file and ConfigClass).")
    
    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]


class BaselineMPCData(BaseMPCData):
    """Data class for Baseline MPC"""

    # files and paths
    results_suffix: str = "_base.csv"
    name_of_created_file: str = "baseline.json"
    # modules
    module_types: dict = cmng.BASELINE_MODULE_TYPE_DICT
    class_name: str = "BaselineMPCModel"
    module_id: str = "Baseline"
    agent_id: str = "Baseline"
    # variables
    power_alias: str = glbs.POWER_ALIAS_BASE
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_BASE
    power_variable: str = pydantic.Field(
        default="P_el",
        description=(
            "Name of the variable representing the electrical "
            "power in the baseline config"
        ),
    )
    profile_deviation_weight: float = pydantic.Field(
        default=0,
        description="Weight of soft constraint for deviation from "
                    "accepted flexible profile",
    )
    power_unit: str = pydantic.Field(
        default="kW", description="Unit of the power variable"
    )
    comfort_variable: str = pydantic.Field(
        default=None,
        description=(
            "Name of the slack variable representing the thermal "
            "comfort in the baseline config"
        ),
    )
    profile_comfort_weight: float = pydantic.Field(
        default=1, description="Weight of soft constraint for discomfort",
    )
    config_inputs_appendix: list[MPCVariable] = [
        MPCVariable(name=glbs.ACCEPTED_POWER_VAR_NAME, value=0, unit="W"),
        MPCVariable(name=glbs.PROVISION_VAR_NAME, value=False),
        MPCVariable(name=glbs.RELATIVE_EVENT_START_TIME_VAR_NAME, value=0, unit="s"),
        MPCVariable(name=glbs.RELATIVE_EVENT_END_TIME_VAR_NAME, value=0, unit="s"),
    ]

    config_parameters_appendix: list[MPCVariable] = []


    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]

    @model_validator(mode="after")
    def update_config_parameters_appendix(self) -> "BaselineMPCData":
        """Update the config parameters appendix with profile deviation and comfort
        weights.

        Adds the profile deviation weight parameter and optionally the profile
        comfort weight parameter (if comfort_variable is enabled) to the
        config_parameters_appendix list as MPCVariable instances.
        """
        self.config_parameters_appendix = [
            MPCVariable(
                name=glbs.PROFILE_DEVIATION_WEIGHT,
                value=self.profile_deviation_weight,
                unit="-",
                description=(
                    "Weight of soft constraint for deviation from accepted "
                    "flexible profile"
                ),
            )
        ]
        if self.comfort_variable:
            self.config_parameters_appendix.append(
                MPCVariable(
                    name=glbs.PROFILE_COMFORT_WEIGHT,
                    value=self.profile_comfort_weight,
                    unit="-",
                    description="Weight of soft constraint for discomfort",
                )
            )
        return self


class PFMPCData(BaseMPCData):
    """Data class for PF-MPC"""

    # files and paths
    results_suffix: str = "_pos_flex.csv"
    name_of_created_file: str = "pos_flex.json"
    # modules
    module_types: dict = cmng.SHADOW_MODULE_TYPE_DICT
    class_name: str = "PosFlexModel"
    module_id: str = "PosFlexMPC"
    agent_id: str = "PosFlexMPC"
    # variables
    power_alias: str = glbs.POWER_ALIAS_POS
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_POS
    flex_cost_function: str = pydantic.Field(
        default=None, description="Cost function of the PF-MPC during the event",
    )
    flex_cost_function_appendix: str = pydantic.Field(
        default=None, description="Cost function appendix of the PF-MPC added "
                                  "to the Baseline cost function",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description="Parameters, which need to be appended to the shadow MPCs"
    )
    config_inputs_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description="Inputs, which need to be appended to the shadow MPCs"
    )
        
    
    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]

    @model_validator(mode="before")
    @classmethod 
    def add_defaults_to_appendix(cls, data): 
        """
        Ensure that all required framework variables are included in both
        ``config_inputs_appendix`` and ``config_parameters_appendix``.
        If any default framework input or parameter (e.g., PROVISION_VAR_NAME)
        is missing, it is appended to the corresponding list.
        """

        data =  _ensure_defaults_in_appendix(
            data=data,
            field_name="config_inputs_appendix",
            default_variables=default_inputs,
        )

        data = _ensure_defaults_in_appendix(
            data=data,
            field_name="config_parameters_appendix",
            default_variables=default_parameters,
        )
        return data

    
class NFMPCData(BaseMPCData):
    """Data class for NF-MPC"""

    # files and paths
    results_suffix: str = "_neg_flex.csv"
    name_of_created_file: str = "neg_flex.json"
    # modules
    module_types: dict = cmng.SHADOW_MODULE_TYPE_DICT
    class_name: str = "NegFlexModel"
    module_id: str = "NegFlexMPC"
    agent_id: str = "NegFlexMPC"
    # variables
    power_alias: str = glbs.POWER_ALIAS_NEG
    stored_energy_alias: str = glbs.STORED_ENERGY_ALIAS_NEG
    flex_cost_function: str = pydantic.Field(
        default=None, description="Cost function of the NF-MPC during the event",
    )
    flex_cost_function_appendix: str = pydantic.Field(
        default=None, description="Cost function appendix of the NF-MPC added "
                                  "to the Baseline cost function",
    )
    # initialize market parameters with dummy values (0)
    config_parameters_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description="Parameters, which need to be appended to the shadow MPCs"
    )
    config_inputs_appendix: list[MPCVariable] = pydantic.Field(
        default=[], description = "Inputs, which need to be appended to the shadow MPCs" 
    )


    @field_serializer('config_inputs_appendix', 'config_parameters_appendix')
    def serialize_mpc_variables(self, variables: list[MPCVariable], _info):
        return [v.dict(exclude=excluded_fields) for v in variables]

    
    @model_validator(mode="before")
    @classmethod 
    def add_defaults_to_appendix(cls, data): 
        """
        Ensures that all required framework input and parameter variables are included in
        ``config_inputs_appendix`` and ``config_parameters_appendix``. If any default
        framework input or parameter (e.g., PROVISION_VAR_NAME) is missing, it is appended
        to the corresponding list.
        """

        data =  _ensure_defaults_in_appendix(
            data=data,
            field_name="config_inputs_appendix",
            default_variables=default_inputs,
        )

        data = _ensure_defaults_in_appendix(
            data=data,
            field_name="config_parameters_appendix",
            default_variables=default_parameters,
        )
        return data
