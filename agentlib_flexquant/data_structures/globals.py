"""Script containing global variables"""

from typing import Literal

# fixed string definitions
PREP_TIME = "prep_time"
MARKET_TIME = "market_time"
FLEX_EVENT_DURATION = "flex_event_duration"
PROFILE_DEVIATION_WEIGHT = "profile_deviation_weight"
PROFILE_COMFORT_WEIGHT = "profile_comfort_weight"
TIME_STEP = "time_step"
PREDICTION_HORIZON = "prediction_horizon"
FLEXIBILITY_OFFER = "FlexibilityOffer"
LINEAR = 'linear'
CONSTANT = 'constant'
COLLOCATION = 'collocation'
INTEGRATION_METHOD = Literal[LINEAR, CONSTANT]
FlexibilityDirections = Literal["positive", "negative"]
POWER_ALIAS_BASE = "_P_el_base"
POWER_ALIAS_NEG = "_P_el_neg"
POWER_ALIAS_POS = "_P_el_pos"
INFLEXIBLE_POWER_ALIAS_BASE = "_P_el_inflexible_base"
INFLEXIBLE_POWER_ALIAS_NEG = "_P_el_inflexible_neg"
INFLEXIBLE_POWER_ALIAS_POS = "_P_el_inflexible_pos"
STORED_ENERGY_ALIAS_BASE = "_E_stored_base"
STORED_ENERGY_ALIAS_NEG = "_E_stored_neg"
STORED_ENERGY_ALIAS_POS = "_E_stored_pos"
full_trajectory_suffix: str = "_full"
base_vars_to_communicate_suffix: str = "_base"
shadow_suffix: str = "_shadow"
COLLOCATION_TIME_GRID = 'collocation_time_grid'
PROVISION_VAR_NAME = "in_provision"
ACCEPTED_POWER_VAR_NAME = "_P_external"
RELATIVE_EVENT_START_TIME_VAR_NAME = "rel_start"
RELATIVE_EVENT_END_TIME_VAR_NAME = "rel_end"
PAUSE_FLEX_CALC = "pause_flex_calc"
OPTIMIZATION_SUCCESS_VAR_NAME = "opti_success"

# cost function in the shadow mpc. obj_std and obj_flex are to be evaluated according
# to user definition
SHADOW_MPC_COST_FUNCTION = (
    "return ca.if_else(self.time < self.prep_time.sym + "
    "self.market_time.sym, obj_std, ca.if_else(self.time < "
    "(self.prep_time.sym + self.flex_event_duration.sym + "
    "self.market_time.sym), obj_flex, obj_std))"
)


def return_baseline_cost_function(power_variable: str, comfort_variable: str) -> str:
    """Return baseline cost function

    Args:
        power_variable: name of the power variable
        comfort_variable: name of the comfort variable

    Returns:
        Cost function in the baseline mpc, obj_std is to be evaluated according to
        user definition

    """
    if comfort_variable:
        cost_func = (
            f"return ca.if_else(self.{PROVISION_VAR_NAME}.sym, "
            f"ca.if_else(self.time < self.{RELATIVE_EVENT_START_TIME_VAR_NAME}.sym, obj_std, "
            f"ca.if_else(self.time >= self.{RELATIVE_EVENT_END_TIME_VAR_NAME}.sym, obj_std, "
            f"sum([self.profile_deviation_weight*(self.{power_variable} - "
            f"self.{ACCEPTED_POWER_VAR_NAME})**2, "
            f"self.{comfort_variable}**2 * self.profile_comfort_weight]))),obj_std)"
        )
    else:
        cost_func = (
            f"return ca.if_else(self.{PROVISION_VAR_NAME}.sym, "
            f"ca.if_else(self.time < self.{RELATIVE_EVENT_START_TIME_VAR_NAME}.sym, obj_std, "
            f"ca.if_else(self.time >= self.{RELATIVE_EVENT_END_TIME_VAR_NAME}.sym, obj_std, "
            f"sum([self.profile_deviation_weight*(self.{power_variable} - "
            f"self.{ACCEPTED_POWER_VAR_NAME})**2]))),obj_std)"
        )
    return cost_func
