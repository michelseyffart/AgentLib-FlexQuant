"""
Flexibility indicator module for calculating and distributing energy flexibility offers.

This module processes power and energy profiles from baseline and shadow MPCs to
calculate flexibility KPIs, validate profile consistency, and generate flexibility
offers for energy markets. It handles both positive and negative flexibility with
optional cost calculations and energy storage corrections.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import agentlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from agentlib_flexquant.utils.data_handling import fill_nans, MEAN

import agentlib_flexquant.data_structures.globals as glbs
from agentlib_flexquant.data_structures.flex_kpis import (
    FlexibilityData,
    FlexibilityKPIs,
)
from agentlib_flexquant.data_structures.flex_offer import FlexOffer


class InputsForCorrectFlexCosts(BaseModel):
    """Configuration for flexibility cost correction.

    """

    enable_energy_costs_correction: bool = Field(
        name="enable_energy_costs_correction",
        description=(
            "Variable determining whether to correct the costs of the "
            "flexible energy. Define the variable for stored electrical "
            "energy in the base MPC model and config as output if the "
            "correction of costs is enabled"
        ),
        default=False,
    )

    absolute_power_deviation_tolerance: float = Field(
        name="absolute_power_deviation_tolerance",
        default=0.1,
        description="Absolute tolerance in kW within which no warning is thrown",
    )

    stored_energy_variable: Optional[str] = Field(
        name="stored_energy_variable",
        default=None,
        description="Name of the variable representing the stored electrical energy "
                    "in the baseline config"
    )


class InputsForCalculateFlexCosts(BaseModel):
    """Configuration for flexibility cost calculation with optional constant
    pricing.

    """

    use_constant_electricity_price: bool = Field(
        default=False, description="Use constant electricity price"
    )
    use_constant_feed_in_price: bool = Field(
        default=False, description="Use constant feed-in price"
    )
    calculate_flex_costs: bool = Field(
        default=True, description="Calculate the flexibility cost"
    )
    const_electricity_price: float = Field(
        default=np.nan, description="constant electricity price in ct/kWh"
    )
    const_feed_in_price: float = Field(
        default=np.nan, description="constant feed-in price in ct/kWh"
    )

    @model_validator(mode="after")
    def validate_constant_prices(self):
        """Validate that valid constant prices are provided when enabled."""

        price_settings = [
            (
                "use_constant_electricity_price",
                "const_electricity_price",
                "electricity",
            ),
            (
                "use_constant_feed_in_price",
                "const_feed_in_price",
                "feed-in",
            ),
        ]

        for use_flag, price_field, label in price_settings:
            if getattr(self, use_flag) and np.isnan(getattr(self, price_field)):
                raise ValueError(
                    (
                        f'Constant {label} price must be a valid float if it is used '
                        f'for calculation. '
                        f'Received "{use_flag}": true, '
                        f'"{price_field}": {getattr(self, price_field)}. '
                        'Please specify them correctly in the "calculate_costs" '
                        "field in the flex config."
                    )
                )

        return self


# Pos and neg kpis to get the right names for plotting
kpis_pos = FlexibilityKPIs(direction="positive")
kpis_neg = FlexibilityKPIs(direction="negative")


class FlexibilityIndicatorModuleConfig(agentlib.BaseModuleConfig):
    """Configuration for flexibility indicator module with power/energy inputs,
    KPI outputs, and cost calculation settings.

    """

    model_config = ConfigDict(extra="forbid")

    inputs: list[agentlib.AgentVariable] = [
        agentlib.AgentVariable(
            name=glbs.POWER_ALIAS_BASE,
            unit="W",
            type="pd.Series",
            description="The power input to the system",
        ),
        agentlib.AgentVariable(
            name=glbs.POWER_ALIAS_NEG,
            unit="W",
            type="pd.Series",
            description="The power input to the system",
        ),
        agentlib.AgentVariable(
            name=glbs.POWER_ALIAS_POS,
            unit="W",
            type="pd.Series",
            description="The power input to the system",
        ),
        agentlib.AgentVariable(
            name=glbs.STORED_ENERGY_ALIAS_BASE,
            unit="kWh",
            type="pd.Series",
            description="Energy stored in the system w.r.t. 0K",
        ),
        agentlib.AgentVariable(
            name=glbs.STORED_ENERGY_ALIAS_NEG,
            unit="kWh",
            type="pd.Series",
            description="Energy stored in the system w.r.t. 0K",
        ),
        agentlib.AgentVariable(
            name=glbs.STORED_ENERGY_ALIAS_POS,
            unit="kWh",
            type="pd.Series",
            description="Energy stored in the system w.r.t. 0K",
        ),
    ]

    outputs: list[agentlib.AgentVariable] = [
        # Flexibility offer
        agentlib.AgentVariable(name=glbs.FLEXIBILITY_OFFER, type="FlexOffer"),
        # Power KPIs
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_full.get_kpi_identifier(),
            unit="W",
            type="pd.Series",
            description="Negative power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_full.get_kpi_identifier(),
            unit="W",
            type="pd.Series",
            description="Positive power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer.get_kpi_identifier(),
            unit="W",
            type="pd.Series",
            description="Negative power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer.get_kpi_identifier(),
            unit="W",
            type="pd.Series",
            description="Positive power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer_min.get_kpi_identifier(),
            unit="W",
            type="float",
            description="Minimum of negative power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer_min.get_kpi_identifier(),
            unit="W",
            type="float",
            description="Minimum of positive power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer_max.get_kpi_identifier(),
            unit="W",
            type="float",
            description="Maximum of negative power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer_max.get_kpi_identifier(),
            unit="W",
            type="float",
            description="Maximum of positive power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_offer_avg.get_kpi_identifier(),
            unit="W",
            type="float",
            description="Average of negative power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_offer_avg.get_kpi_identifier(),
            unit="W",
            type="float",
            description="Average of positive power flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.power_flex_within_boundary.get_kpi_identifier(),
            unit="-",
            type="bool",
            description=(
                "Variable indicating whether the baseline power and flex power "
                "align at the horizon end"
            ),
        ),
        agentlib.AgentVariable(
            name=kpis_pos.power_flex_within_boundary.get_kpi_identifier(),
            unit="-",
            type="bool",
            description=(
                "Variable indicating whether the baseline power and flex power "
                "align at the horizon end"
            ),
        ),
        # Energy KPIs
        agentlib.AgentVariable(
            name=kpis_neg.energy_flex.get_kpi_identifier(),
            unit="kWh",
            type="float",
            description="Negative energy flexibility",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.energy_flex.get_kpi_identifier(),
            unit="kWh",
            type="float",
            description="Positive energy flexibility",
        ),
        # Costs KPIs
        agentlib.AgentVariable(
            name=kpis_neg.costs.get_kpi_identifier(),
            unit="ct",
            type="float",
            description="Saved costs due to baseline",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.costs.get_kpi_identifier(),
            unit="ct",
            type="float",
            description="Saved costs due to baseline",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.corrected_costs.get_kpi_identifier(),
            unit="ct",
            type="float",
            description="Corrected saved costs due to baseline",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.corrected_costs.get_kpi_identifier(),
            unit="ct",
            type="float",
            description="Corrected saved costs due to baseline",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.costs_rel.get_kpi_identifier(),
            unit="ct/kWh",
            type="float",
            description="Saved costs due to baseline",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.costs_rel.get_kpi_identifier(),
            unit="ct/kWh",
            type="float",
            description="Saved costs due to baseline",
        ),
        agentlib.AgentVariable(
            name=kpis_neg.corrected_costs_rel.get_kpi_identifier(),
            unit="ct/kWh",
            type="float",
            description="Corrected saved costs per energy due to baseline",
        ),
        agentlib.AgentVariable(
            name=kpis_pos.corrected_costs_rel.get_kpi_identifier(),
            unit="ct/kWh",
            type="float",
            description="Corrected saved costs per energy due to baseline",
        ),
    ]

    parameters: list[agentlib.AgentVariable] = [
        agentlib.AgentVariable(name=glbs.PREP_TIME, unit="s",
                               description="Preparation time"),
        agentlib.AgentVariable(name=glbs.MARKET_TIME, unit="s",
                               description="Market time"),
        agentlib.AgentVariable(name=glbs.FLEX_EVENT_DURATION, unit="s",
                               description="time to switch objective"),
        agentlib.AgentVariable(name=glbs.TIME_STEP, unit="s",
                               description="timestep of the mpc solution"),
        agentlib.AgentVariable(name=glbs.PREDICTION_HORIZON, unit="-",
                               description="prediction horizon of the mpc solution"),
        agentlib.AgentVariable(name=glbs.COLLOCATION_TIME_GRID,
                               alias=glbs.COLLOCATION_TIME_GRID,
                               description="Time grid of the mpc model output")
    ]

    results_file: Optional[Path] = Field(
        default=Path("flexibility_indicator.csv"),
        description="User specified results file name",
    )
    save_results: Optional[bool] = Field(
        validate_default=True,
        default=True
    )
    price_variable: str = Field(
        default="c_pel", description="Name of the price variable sent by a predictor",
    )
    price_variable_feed_in: str = Field(
        default="c_pel_feed_in",
        description="Name of the feed-in price variable sent by a predictor",
    )
    power_unit: str = Field(
        default="kW",
        description="Unit of the power variable"
    )
    integration_method: glbs.INTEGRATION_METHOD = Field(
        default=glbs.LINEAR,
        description="Method set to integrate series variable"
    )
    shared_variable_fields: list[str] = ["outputs"]

    correct_costs: InputsForCorrectFlexCosts = InputsForCorrectFlexCosts()
    calculate_costs: InputsForCalculateFlexCosts = InputsForCalculateFlexCosts()

    @model_validator(mode="after")
    def check_results_file_extension(self):
        """Validate that results_file has a .csv extension."""
        if self.results_file and self.results_file.suffix != ".csv":
            raise ValueError(
                f"Invalid file extension for 'results_file': '{self.results_file}'. "
                f"Expected a '.csv' file."
            )
        return self

class CallBackHandler: 
    """Helper class to manage callback handling for flexibility indicator module.
    
    Adapter, der self.data schreibt 

    """
    necessary_callback_variables: dict[str,dict[str, bool]]

    def __init__(self,config: FlexibilityIndicatorModuleConfig): 
        """Load general settings"""
        # set collocation time grid 
        def get_param(cfg, name: str):
            return next(v for v in cfg.parameters if v.name == name)
        self.collocation_time_grid = get_param(config, glbs.COLLOCATION_TIME_GRID).value
        self.necessary_callback_variables = {
            glbs.POWER_ALIAS_BASE: {"name":"power_profile_base", "is_mpc":True},
            glbs.POWER_ALIAS_NEG: {"name":"power_profile_flex_neg", "is_mpc":True},
            glbs.POWER_ALIAS_POS: {"name":"power_profile_flex_pos", "is_mpc":True},
            }

    def update_price_variables(self, config: FlexibilityIndicatorModuleConfig, data: FlexibilityData):
        if config.calculate_costs.calculate_flex_costs:
            if config.calculate_costs.use_constant_electricity_price:
                electricity_price_series = pd.Series(
                    data=config.calculate_costs.const_electricity_price,
                    index=data.mpc_time_grid,
                )
                data.update_profile("electricity_price_series", electricity_price_series, mpc=False)
            else:
                self.necessary_callback_variables.update({config.price_variable: {"name":"electricity_price_series", "is_mpc":False}})
            
            if config.calculate_costs.use_constant_feed_in_price:
                feed_in_price_series = pd.Series(
                    data=config.calculate_costs.const_feed_in_price,
                    index=data.mpc_time_grid,
                )
                data.update_profile("feed_in_price_series", feed_in_price_series, mpc=False)
            else:
                self.necessary_callback_variables.update({config.price_variable_feed_in: {"name":"feed_in_price_series", "is_mpc":False}})
        return data
    
    def initialize_callback_variables(self, data: FlexibilityData, config: FlexibilityIndicatorModuleConfig) -> FlexibilityData:
        data = self.update_price_variables(config=config, data=data)
        if config.correct_costs.enable_energy_costs_correction:
            self.necessary_callback_variables.update({
                glbs.STORED_ENERGY_ALIAS_BASE: {"name":"stored_energy_profile_base", "is_mpc":True},
                glbs.STORED_ENERGY_ALIAS_NEG: {"name":"stored_energy_profile_flex_neg", "is_mpc":True},
                glbs.STORED_ENERGY_ALIAS_POS: {"name":"stored_energy_profile_flex_pos", "is_mpc":True},
            })
            
        return data
    
    def set_all_callback_variables_to_none(self, data: FlexibilityData) -> FlexibilityData:
        """Clear the values of the callback variables after processing."""
        for alias, var  in self.necessary_callback_variables.items():
            data.update_profile(var["name"], None, mpc=var["is_mpc"])
        return data

    def update_input(self, data: FlexibilityData, name: str, value: pd.Series) -> FlexibilityData: 
        """Update the incoming value"""
        var_tuple = self.necessary_callback_variables.get(name, None)
        if var_tuple is not None: 
            variable_name, mpc = var_tuple["name"], var_tuple["is_mpc"]
            data.update_profile(variable_name, value, mpc=mpc)
        return data
    
    def is_ready_for_calculation(self, data: FlexibilityData) -> bool:
        """Check if all necessary profiles and parameters are set for KPI calculation."""
        required_profiles = [getattr(data, var["name"]) for key, var in self.necessary_callback_variables.items()]
        return all(profile is not None for profile in required_profiles)

class FlexibilityIndicatorModule(agentlib.BaseModule):
    """Module for calculating flexibility KPIs and generating flexibility offers
    from MPC power/energy profiles."""

    config: FlexibilityIndicatorModuleConfig
    data: FlexibilityData
    callback_handler: CallBackHandler

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_list = []
        for variable in self.variables:
            if variable.name in [glbs.FLEXIBILITY_OFFER]:
                continue
            self.var_list.append(variable.name)
        self.time = []
        self.in_provision = False
        self.offer_count = 0
        self.df = pd.DataFrame(columns=pd.Series(self.var_list))
        self.data = FlexibilityData(
            prep_time=self.get(glbs.PREP_TIME).value,
            market_time=self.get(glbs.MARKET_TIME).value,
            flex_event_duration=self.get(glbs.FLEX_EVENT_DURATION).value,
            time_step=self.get(glbs.TIME_STEP).value,
            prediction_horizon=self.get(glbs.PREDICTION_HORIZON).value,
        )
        self.callback_handler = CallBackHandler(config=self.config)
        self.data = self.callback_handler.initialize_callback_variables(data=self.data, config=self.config)

    def register_callbacks(self):
        inputs = self.config.inputs
        for var in inputs:
            self.agent.data_broker.register_callback(
                name=var.name, alias=var.name, callback=self.callback
            )
        self.agent.data_broker.register_callback(
            name=glbs.PROVISION_VAR_NAME, alias=glbs.PROVISION_VAR_NAME,
            callback=self.callback
        )

    def process(self):
        """Yield control to the simulation environment and wait for events."""
        yield self.env.event()

    def callback(self, inp, name):
        """Handle incoming data by storing power/energy/price profiles and triggering
        flexibility calculations when all required inputs are available.
        """ 
        
        if name == glbs.PROVISION_VAR_NAME:
            self.in_provision = inp.value

        if self.in_provision:
            self.data = self.callback_handler.set_all_callback_variables_to_none(data=self.data)
        else: 
            self.data = self.callback_handler.update_input(data=self.data, name=name, value=inp.value)
            
        if self.callback_handler.is_ready_for_calculation(data=self.data):
            # check the power profile end deviation
            if not self.config.correct_costs.enable_energy_costs_correction:
                self.check_power_end_deviation(
                    tol=self.config.correct_costs.absolute_power_deviation_tolerance
                )
            # calculate and send the offer and reset the callback variables 
            self.calc_and_send_offer()
            self.data = self.callback_handler.set_all_callback_variables_to_none(data=self.data)
        
    def get_results(self) -> Optional[pd.DataFrame]:
        """Open results file of flexibility_indicator.py."""
        results_file = self.config.results_file
        try:
            results = pd.read_csv(results_file, header=[0], index_col=[0, 1])
            return results
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

    def write_results(self, df: pd.DataFrame, ts: float, n: int) -> pd.DataFrame:
        """Write every data of variables in self.var_list in an DataFrame.

        DataFrame will be updated every time step

        Args:
            df: DataFrame which is initialised as an empty DataFrame with columns
            according to self.var_list
            ts: time step
            n: number of time steps during prediction horizon

        Returns:
            DataFrame with results of every variable in self.var_list

        """
        results = []
        now = self.env.now

        # First, collect all series and their indices
        all_series = []
        for name in self.var_list:
            # Get the appropriate values based on name
            if name == glbs.POWER_ALIAS_BASE:
                values = self.data.power_profile_base
            elif name == glbs.POWER_ALIAS_NEG:
                values = self.data.power_profile_flex_neg
            elif name == glbs.POWER_ALIAS_POS:
                values = self.data.power_profile_flex_pos
            elif name == glbs.STORED_ENERGY_ALIAS_BASE:
                values = self.data.stored_energy_profile_base
            elif name == glbs.STORED_ENERGY_ALIAS_NEG:
                values = self.data.stored_energy_profile_flex_neg
            elif name == glbs.STORED_ENERGY_ALIAS_POS:
                values = self.data.stored_energy_profile_flex_pos
            elif name == self.config.price_variable:
                values = self.data.electricity_price_series
            elif name == self.config.price_variable_feed_in:
                values = self.data.feed_in_price_series
            elif name == glbs.COLLOCATION_TIME_GRID:
                value = self.get(name).value
                values = pd.Series(index=value, data=value)
            else:
                values = self.get(name).value

            # Convert to Series if not already
            if not isinstance(values, pd.Series):
                values = pd.Series(values)

            all_series.append((name, values))

        # Create the standard grid for reference
        standard_grid = np.arange(0, n * ts, ts)

        # Find the union of all indices to create a comprehensive grid
        all_indices = set(standard_grid)
        for _, series in all_series:
            all_indices.update(series.index)
        combined_index = sorted(all_indices)

        # Reindex all series to the combined grid
        for i, (name, series) in enumerate(all_series):
            # Reindex to the comprehensive grid
            reindexed = series.reindex(combined_index)
            results.append(reindexed)

        if not now % ts:
            self.time.append(now)
            new_df = pd.DataFrame(results).T
            new_df.columns = self.var_list
            # Rename time_step variable column
            new_df.rename(
                columns={glbs.TIME_STEP: f"{glbs.TIME_STEP}_mpc"}, inplace=True
            )
            new_df.index.direction = "time"
            new_df[glbs.TIME_STEP] = now
            new_df.set_index([glbs.TIME_STEP, new_df.index], inplace=True)
            df = pd.concat([df, new_df])
            # set the indices once again as concat cant handle indices properly
            indices = pd.MultiIndex.from_tuples(
                df.index, names=[glbs.TIME_STEP, "time"]
            )
            df.set_index(indices, inplace=True)
            # Drop column time_step and keep it as an index only
            if glbs.TIME_STEP in df.columns:
                df.drop(columns=[glbs.TIME_STEP], inplace=True)

        return df

    def cleanup_results(self):
        """Remove the existing result files."""
        results_file = self.config.results_file
        if not results_file:
            return
        os.remove(results_file)

    def calc_and_send_offer(self):
        """Calculate the flexibility KPIs for current predictions, send the flex offer
        and set the outputs, write and save the results."""
        # Calculate the flexibility KPIs for current predictions
        collocation_time_grid = self.get(glbs.COLLOCATION_TIME_GRID).value
        self.data.calculate(
            enable_energy_costs_correction=
            self.config.correct_costs.enable_energy_costs_correction,
            calculate_flex_cost=self.config.calculate_costs.calculate_flex_costs,
            integration_method=self.config.integration_method,
            collocation_time_grid=collocation_time_grid)

        # get the full index during flex event including mpc_time_grid index and the
        # collocation index
        full_index = np.sort(np.concatenate([collocation_time_grid,
                                             self.data.mpc_time_grid]))
        flex_begin = self.get(glbs.MARKET_TIME).value + self.get(glbs.PREP_TIME).value
        flex_end = flex_begin + self.get(glbs.FLEX_EVENT_DURATION).value
        full_flex_offer_index = full_index[(full_index >= flex_begin) &
                                           (full_index <= flex_end)]

        # reindex the power profiles to not send the simulation points to the market,
        # but only the values on the collocation points and the forward mean of them
        base_power_profile = self.data.power_profile_base.reindex(
            collocation_time_grid).reindex(full_flex_offer_index)
        pos_diff_profile = self.data.kpis_pos.power_flex_offer.value.reindex(
            collocation_time_grid).reindex(full_flex_offer_index)
        neg_diff_profile = self.data.kpis_neg.power_flex_offer.value.reindex(
            collocation_time_grid).reindex(full_flex_offer_index)

        # fill the mpc_time_grid with forward mean
        base_power_profile = fill_nans(base_power_profile, method=MEAN)
        pos_diff_profile = fill_nans(pos_diff_profile, method=MEAN)
        neg_diff_profile = fill_nans(neg_diff_profile, method=MEAN)

        # Send flex offer
        self.send_flex_offer(
            name=glbs.FLEXIBILITY_OFFER,
            base_power_profile=base_power_profile,
            pos_diff_profile=pos_diff_profile,
            pos_price=self.data.kpis_pos.costs.value,
            neg_diff_profile=neg_diff_profile,
            neg_price=self.data.kpis_neg.costs.value,
        )

        # set outputs
        for kpi in self.data.get_kpis().values():
            if kpi.get_kpi_identifier() not in [
                kpis_pos.power_flex_within_boundary.get_kpi_identifier(),
                kpis_neg.power_flex_within_boundary.get_kpi_identifier(),
            ]:
                for output in self.config.outputs:
                    if output.name == kpi.get_kpi_identifier():
                        self.set(output.name, kpi.value)

        # write results
        self.df = self.write_results(
            df=self.df,
            ts=self.get(glbs.TIME_STEP).value,
            n=self.get(glbs.PREDICTION_HORIZON).value,
        )

        # save results
        if self.config.save_results:
            self.df.to_csv(self.config.results_file)

    def send_flex_offer(
        self,
        name: str,
        base_power_profile: pd.Series,
        pos_diff_profile: pd.Series,
        pos_price: float,
        neg_diff_profile: pd.Series,
        neg_price: float,
        timestamp: float = None,
    ):
        """Send a flex offer as an agent Variable.

        The first offer is dismissed, since the different MPCs need one time step
        to fully initialize.

        Args:
            name: name of the agent variable
            base_power_profile: time series of power from baseline mpc
            pos_diff_profile: power profile for the positive difference (base-pos)
            in flexibility event time grid
            pos_price: price for positive flexibility
            neg_diff_profile: power profile for the negative difference (neg-base)
            in flexibility event time grid
            neg_price: price for negative flexibility
            timestamp: the time offer was generated

        """
        if self.offer_count > 0:
            var = self._variables_dict[name]
            var.value = FlexOffer(
                base_power_profile=base_power_profile,
                pos_diff_profile=pos_diff_profile,
                pos_price=pos_price,
                neg_diff_profile=neg_diff_profile,
                neg_price=neg_price,
            )
            if timestamp is None:
                timestamp = self.env.time
            var.timestamp = timestamp
            self.agent.data_broker.send_variable(
                variable=var.copy(update={"source": self.source}), copy=False,
            )
        self.offer_count += 1

    def check_power_end_deviation(self, tol: float):
        """Calculate the deviation of the final value of the power profiles
        and warn the user if it exceeds the tolerance."""
        logger = logging.getLogger(__name__)
        dev_pos = np.mean(
            self.data.power_profile_flex_pos.values[-4:]
            - self.data.power_profile_base.values[-4:]
        )
        dev_neg = np.mean(
            self.data.power_profile_flex_neg.values[-4:]
            - self.data.power_profile_base.values[-4:]
        )
        if abs(dev_pos) > tol:
            logger.warning(
                "There is an average deviation of %.6f kW between the final values of "
                "power profiles of positive shadow MPC and the baseline. "
                "Correction of energy costs might be necessary.",
                dev_pos,
            )
            self.set(kpis_pos.power_flex_within_boundary.get_kpi_identifier(), False)
        else:
            self.set(kpis_pos.power_flex_within_boundary.get_kpi_identifier(), True)
        if abs(dev_neg) > tol:
            logger.warning(
                "There is an average deviation of %.6f kW between the final values of "
                "power profiles of negative shadow MPC and the baseline. "
                "Correction of energy costs might be necessary.",
                dev_neg,
            )
            self.set(kpis_neg.power_flex_within_boundary.get_kpi_identifier(), False)
        else:
            self.set(kpis_neg.power_flex_within_boundary.get_kpi_identifier(), True)



