import logging
import os
from pathlib import Path

import casadi as ca

from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)


# script variables
ub = 295.15

# constants
COOLING = 1000


class MyCasadiModelConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        # controls
        CasadiInput(
            name="cooling_power",
            value=400,
            unit="W",
            description="Air mass flow into zone",
        ),
        CasadiInput(
            name="cooler_on",
            value=1,
            unit="-",
            description="On / off signal of mass flow.",
            lb=0,
            ub=1,
        ),
        # disturbances
        CasadiInput(
            name="load", value=150, unit="W", description="Heat load into zone"
        ),
        CasadiInput(
            name="T_in", value=290.15, unit="K", description="Inflow air temperature"
        ),
        # settings
        CasadiInput(
            name="T_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
    ]

    states: list[CasadiState] = [
        # differential
        CasadiState(
            name="T", value=293.15, unit="K", description="Temperature of zone"
        ),
        # algebraic
        # slack variables
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone",
        ),
    ]

    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="C", value=100000, unit="J/K", description="thermal capacity of zone"
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for T in constraint function",
        ),
        CasadiParameter(
            name="r_cooling",
            value=1 / 5,
            unit="-",
            description="Weight for mDot in objective function",
        ),
        CasadiParameter(
            name="cooler_mod_limit",
            value=200,
            unit="W",
            description="Cooling power cannot modulate below this value",
        ),
    ]
    outputs: list[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone"),
        CasadiOutput(name="P_el", unit="W", description="Electrical power")
    ]


class MyCasadiModel(CasadiModel):

    config: MyCasadiModelConfig

    def setup_system(self):
        # Define ode
        self.T.ode = (self.load - self.cooling_power) / self.C

        # Define ae
        self.T_out.alg = self.T  # math operation to get the symbolic variable
        self.P_el.alg = self.cooling_power  # math operation to get the symbolic variable

        # Constraints: list[(lower bound, function, upper bound)]
        self.constraints = [
            # bigM reformulation
            (-ca.inf, self.cooling_power - self.cooler_on * COOLING, 0),
            (0, self.cooling_power - self.cooler_on * self.cooler_mod_limit, ca.inf),
            # soft constraints
            (0, self.T + self.T_slack, self.T_upper),
        ]

        # Objective function
        objective = sum(
            [
                self.r_cooling * self.cooling_power,
                self.s_T * self.T_slack**2,
            ]
        )

        return objective
