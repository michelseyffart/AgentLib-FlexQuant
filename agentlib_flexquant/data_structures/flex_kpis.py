"""
Module for representing and calculating flexibility KPIs. It defines Pydantic models
for scalar and time-series KPIs, and provides methods to compute power, energy,
and cost metrics for positive and negative flexibility scenarios.
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
import pydantic
from agentlib_mpc.utils import TIME_CONVERSION, TimeConversionTypes

from agentlib_flexquant.data_structures.globals import (
    FlexibilityDirections,
    LINEAR,
    CONSTANT,
    INTEGRATION_METHOD,
)
from agentlib_flexquant.utils.data_handling import MEAN, fill_nans, strip_multi_index


class KPI(pydantic.BaseModel):
    """Class defining attributes of the indicator KPI."""

    name: str = pydantic.Field(
        default=None,
        description="Name of the flexibility KPI",
    )
    value: Union[float, None] = pydantic.Field(
        default=None,
        description="Value of the flexibility KPI",
    )
    unit: str = pydantic.Field(
        default=None,
        description="Unit of the flexibility KPI",
    )
    direction: Union[FlexibilityDirections, None] = pydantic.Field(
        default=None, description="Direction of the shadow mpc / flexibility"
    )

    class Config:
        """Allow arbitrary (non-Pydantic) types such as pandas.Series or numpy.ndarray
        in model fields without requiring custom validators."""

        arbitrary_types_allowed = True

    def get_kpi_identifier(self):
        """Get the identifier of the KPI composed of the direction of the flexibility
        and the KPI name."""
        name = f"{self.direction}_{self.name}"
        return name
    

class KPISeries(KPI):
    """Class defining extra attributes of the indicator KPISeries in addition to KPI."""

    value: Union[pd.Series, None] = pydantic.Field(
        default=None,
        description="Value of the flexibility KPI",
    )
    dt: Union[pd.Series, None] = pydantic.Field(
        default=None,
        description="Time differences between the timestamps of the series in seconds",
    )
    integration_method: INTEGRATION_METHOD = pydantic.Field(
        default=LINEAR, description="Method set to integrate series variable"
    )

    def _get_dt(self) -> pd.Series:
        """Get the forward time differences between the timestamps of the series."""
        # compute forward differences between consecutive timestamps
        dt = pd.Series(index=self.value.index, data=self.value.index).diff().shift(-1)
        # set the last value of dt to zero since there is no subsequent time step to compute a
        # difference with
        dt.iloc[-1] = 0
        self.dt = dt
        return dt

    def min(self) -> float:
        """Get the minimum of a KPISeries."""
        return self.value.min()

    def max(self) -> float:
        """Get the maximum of a KPISeries."""
        return self.value.max()

    def avg(self) -> float:
        """Calculate the average value of the KPISeries over time."""
        if self.dt is None:
            self._get_dt()
        delta_t = self.dt.sum()
        avg = self.integrate() / delta_t
        return avg

    def integrate(self, time_unit: TimeConversionTypes = "seconds") -> float:
        """Integrate the value of the KPISeries over time by summing up
        the product of values and the time difference.

        Args:
            time_unit: The time unit the integrated value should have

        Returns:
            The integrated value of the KPISeries

        """
        if self.integration_method == LINEAR:
            # Linear integration: apply the trapezoidal rule, which assumes
            # the function changes linearly between sample points
            return np.trapezoid(self.value.values,
                                self.value.index) / TIME_CONVERSION[time_unit]
        if self.integration_method == CONSTANT:
            # Constant integration: use a step-wise constant approach by
            # holding the value constant over each interval
            return (
                np.sum(self.value.values[:-1] * self._get_dt().iloc[:-1])
                / TIME_CONVERSION[time_unit]
            )
        

class FlexibilityKPIs(pydantic.BaseModel):
    """Class defining the indicator KPIs."""

    # Direction
    direction: FlexibilityDirections = pydantic.Field(
        default=None, description="Direction of the shadow mpc"
    )

    # Power / energy KPIs
    power_flex_full: KPISeries = pydantic.Field(
        default=KPISeries(name="power_flex_full", unit="kW", integration_method=LINEAR),
        description="Power flexibility",
    )
    power_flex_offer: KPISeries = pydantic.Field(
        default=KPISeries(name="power_flex_offer", unit="kW", integration_method=LINEAR),
        description="Power flexibility",
    )
    power_flex_offer_max: KPI = pydantic.Field(
        default=KPI(name="power_flex_offer_max", unit="kW"),
        description="Maximum power flexibility",
    )
    power_flex_offer_min: KPI = pydantic.Field(
        default=KPI(name="power_flex_offer_min", unit="kW"),
        description="Minimum power flexibility",
    )
    power_flex_offer_avg: KPI = pydantic.Field(
        default=KPI(name="power_flex_offer_avg", unit="kW"),
        description="Average power flexibility",
    )
    energy_flex: KPI = pydantic.Field(
        default=KPI(name="energy_flex", unit="kWh"),
        description="Energy flexibility equals the integral of the power flexibility",
    )
    power_flex_within_boundary: KPI = pydantic.Field(
        default=KPI(name="power_flex_within_boundary", unit="-"),
        description=(
            "Variable indicating whether the baseline power and flex power "
            "align at the horizon end"
        ),
    )

    # Costs KPIs
    electricity_costs_series: KPISeries = pydantic.Field(
        default=KPISeries(name="electricity_costs_series", unit="ct/h", integration_method=LINEAR),
        description="Difference in electricity costs between shadow and baseline mpc over full prediction horizon",
    )
    costs: KPI = pydantic.Field(
        default=KPI(name="costs", unit="ct"),
        description="Costs of flexibility",
    )
    corrected_costs: KPI = pydantic.Field(
        default=KPI(name="corrected_costs", unit="ct"),
        description="Corrected costs of flexibility considering the stored "
                    "energy in the system",
    )
    costs_rel: KPI = pydantic.Field(
        default=KPI(name="costs_rel", unit="ct/kWh"),
        description="Costs of flexibility per energy",
    )
    corrected_costs_rel: KPI = pydantic.Field(
        default=KPI(name="corrected_costs_rel", unit="ct/kWh"),
        description="Corrected costs of flexibility per energy",
    )

    def __init__(self, direction: FlexibilityDirections, **data):
        super().__init__(**data)
        self.direction = direction
        for kpi in vars(self).values():
            if isinstance(kpi, KPI):
                kpi.direction = self.direction

    def calculate(
        self,
        power_profile_base: pd.Series,
        power_profile_shadow: pd.Series,
        electricity_price_series: pd.Series,
        feed_in_price_series: pd.Series,
        mpc_time_grid: np.ndarray,
        flex_offer_time_grid: np.ndarray,
        stored_energy_base: pd.Series,
        stored_energy_shadow: pd.Series,
        enable_energy_costs_correction: bool,
        calculate_flex_cost: bool,
        integration_method: INTEGRATION_METHOD,
        collocation_time_grid: list = None,
    ):
        """Calculate the KPIs based on the power and electricity price input profiles.

        Args:
            power_profile_base: power profile from baseline mpc
            power_profile_shadow: power profile from shadow mpc
            electricity_price_series: time series of electricity prices
            feed_in_price_series: time series of electricity feed-in prices
            flex_offer_time_grid: time grid over which the flexibility offer is calculated,
            for indexing of the power flexibility profiles
            stored_energy_base: time series of stored energy from baseline mpc
            stored_energy_shadow: time series of stored energy from shadow mpc
            enable_energy_costs_correction: whether the energy costs should be corrected
            calculate_flex_cost: whether the cost of the flexibility should be calculated
            integration_method: method used for integration of KPISeries e.g. linear, constant
            collocation_time_grid: Time grid of the mpc output with collocation discretization


        """
        # Power / energy KPIs
        self._calculate_power_flex(
            power_profile_base=power_profile_base,
            power_profile_shadow=power_profile_shadow,
            flex_offer_time_grid=flex_offer_time_grid,
            integration_method=integration_method,
        )
        self._calculate_power_flex_stats(
            mpc_time_grid=mpc_time_grid, collocation_time_grid=collocation_time_grid
        )
        self._calculate_energy_flex(
            mpc_time_grid=mpc_time_grid, collocation_time_grid=collocation_time_grid
        )

        # Costs KPIs
        if enable_energy_costs_correction:
            stored_energy_diff = stored_energy_shadow.values[-1] - stored_energy_base.values[-1]
        else:
            stored_energy_diff = 0

        if calculate_flex_cost:
            self._calculate_costs(
                electricity_price_signal=electricity_price_series,
                feed_in_price_signal=feed_in_price_series,
                power_profile_base=power_profile_base,
                power_profile_shadow=power_profile_shadow,
                stored_energy_diff=stored_energy_diff,
                integration_method=integration_method,
                mpc_time_grid=mpc_time_grid,
                collocation_time_grid=collocation_time_grid,
            )
            self._calculate_costs_rel()

    def _calculate_power_flex(
        self,
        power_profile_base: pd.Series,
        power_profile_shadow: pd.Series,
        flex_offer_time_grid: np.ndarray,
        integration_method: INTEGRATION_METHOD,
        relative_error_acceptance: float = 0.01,
    ):
        """Calculate the power flexibility based on the base and flexibility power profiles.

        Args:
            power_profile_base: power profile from the baseline mpc
            power_profile_shadow: power profile from the shadow mpc
            flex_offer_time_grid: time grid over which the flexibility offer is calculated
            integration_method: method used for integration of KPISeries e.g. linear, constant
            relative_error_acceptance: threshold for the relative error between the baseline
            and shadow mpc to set the power flexibility to zero

        """
        if not power_profile_shadow.index.equals(power_profile_base.index):
            raise ValueError(
                f"Indices of power profiles do not match.\n"
                f"Baseline: {power_profile_base.index}\n"
                f"Shadow: {power_profile_shadow.index}"
            )

        # Calculate power flexibility trajectory
        if self.direction == "positive":
            power_flex = power_profile_base - power_profile_shadow
        elif self.direction == "negative":
            power_flex = power_profile_shadow - power_profile_base
        else:
            raise ValueError(
                f"Direction of KPIs not properly defined: {self.direction}")

        # Set values to zero if the difference is small
        relative_difference = (power_flex / power_profile_base).abs()
        power_flex.loc[relative_difference < relative_error_acceptance] = 0
        # Set the first value of power_flex to zero, since it comes from the measurement/simulator
        # and is the same for baseline and shadow mpcs.
        # For quantification of flexibility, only power difference is of interest.
        power_flex.iloc[0] = 0

        # Set values
        self.power_flex_full.value = power_flex
        self.power_flex_offer.value = power_flex.loc[
            flex_offer_time_grid[0] : flex_offer_time_grid[-1]
        ]

        # Set integration method
        self.power_flex_full.integration_method = integration_method
        self.power_flex_offer.integration_method = integration_method

    def _calculate_power_flex_stats(
        self, mpc_time_grid: np.array, collocation_time_grid: list = None
    ):
        """Calculate the characteristic values of the power flexibility for the offer."""
        if self.power_flex_offer.value is None:
            raise ValueError("Power flexibility value is empty.")

        # Calculate characteristic values
        # max and min of power flex offer
        power_flex_offer = self.power_flex_offer.value.iloc[:-1].drop(
            collocation_time_grid, errors="ignore"
        )
        power_flex_offer_max = power_flex_offer.max()
        power_flex_offer_min = power_flex_offer.min()
        # Average of the power flex offer
        # Get the series for integration before calculating average
        power_flex_offer_integration = self._get_series_for_integration(
            series=self.power_flex_offer, mpc_time_grid=mpc_time_grid
        )
        power_flex_offer_integration.value = power_flex_offer_integration.value.drop(
            collocation_time_grid, errors="ignore"
        )
        # Calculate the average and stores the original value
        power_flex_offer_avg = power_flex_offer_integration.avg()

        # Set values
        self.power_flex_offer_max.value = power_flex_offer_max
        self.power_flex_offer_min.value = power_flex_offer_min
        self.power_flex_offer_avg.value = power_flex_offer_avg

    def _get_series_for_integration(
        self, series: KPISeries, mpc_time_grid: np.ndarray
    ) -> KPISeries:
        """Return the KPISeries value sampled on the MPC time grid when the integration
        method is constant.

        Otherwise, the original value is returned.

        Args:
            series: the KPISeries to get value from
            mpc_time_grid: the MPC time grid over the horizon

        """
        if series.integration_method == CONSTANT:
            series = series.__deepcopy__()
            series.value = series.value.reindex(mpc_time_grid).dropna()
            return series
        else:
            return series.__deepcopy__()

    def _calculate_energy_flex(self, mpc_time_grid, collocation_time_grid: list = None):
        """Calculate the energy flexibility by integrating the power flexibility
        of the offer window."""
        if self.power_flex_offer.value is None:
            raise ValueError("Power flexibility value of the offer is empty.")

        # Calculate flexibility
        # Get the series for integration before calculating average
        power_flex_offer_integration = self._get_series_for_integration(
            series=self.power_flex_offer, mpc_time_grid=mpc_time_grid
        )
        power_flex_offer_integration.value = power_flex_offer_integration.value.drop(
            collocation_time_grid, errors="ignore"
        )
        # Calculate the energy flex and stores the original value
        energy_flex = power_flex_offer_integration.integrate(time_unit="hours")

        # Set value
        self.energy_flex.value = energy_flex

    def _calculate_costs(
        self,
        electricity_price_signal: pd.Series,
        feed_in_price_signal: pd.Series,
        power_profile_base: pd.Series,
        power_profile_shadow: pd.Series,
        stored_energy_diff: float,
        integration_method: INTEGRATION_METHOD,
        mpc_time_grid: np.ndarray,
        collocation_time_grid: list = None,
    ):
        """Calculate the costs of the flexibility event based on the electricity costs profile,
        the power flexibility profile and difference of stored energy.

        Args:
            electricity_price_signal: time series of the electricity price signal
            feed_in_price_signal: time series of the feed-in price signal
            power_profile_base: baseline power profile used to select tariff by sign
            power_profile_shadow: shadow mpc power profile used to select tariff by sign
            stored_energy_diff: the difference of the stored energy between baseline and shadow mpc
            integration_method: the integration method used to integrate KPISeries
            mpc_time_grid: the MPC time grid over the horizon
            collocation_time_grid: Time grid of the mpc output with collocation discretization


        """
        if not power_profile_shadow.index.equals(power_profile_base.index):
            raise ValueError(
                f"Indices of power profiles do not match.\n"
                f"Baseline: {power_profile_base.index}\n"
                f"Shadow: {power_profile_shadow.index}"
            )

        # Set integration method
        self.power_flex_full.integration_method = integration_method
        self.electricity_costs_series.integration_method = integration_method

        # if there is no feed-in tariff provided, the electricity price signal is used for both consumption and feed-in
        if feed_in_price_signal is None:
            feed_in_price_signal = electricity_price_signal

        # Select tariff based on the sign of each profile
        effective_price_base = electricity_price_signal.where(
            power_profile_base > 0,
            feed_in_price_signal,
        )
        effective_price_shadow = electricity_price_signal.where(
            power_profile_shadow > 0,
            feed_in_price_signal,
        )

        cost_profile_base = power_profile_base * effective_price_base
        cost_profile_shadow = power_profile_shadow * effective_price_shadow

        # Get the series for integration before calculating
        power_flex_full_integration = self._get_series_for_integration(
            series=self.power_flex_full, mpc_time_grid=mpc_time_grid
        )
        power_flex_full_integration.value = power_flex_full_integration.value.drop(
            collocation_time_grid, errors="ignore"
        )

        # Difference in costs between shadow and baseline mpc
        delta_cost = cost_profile_shadow - cost_profile_base
        delta_cost = delta_cost.reindex(power_flex_full_integration.value.index)
        delta_cost = delta_cost.where(power_flex_full_integration.value != 0, 0)
        self.electricity_costs_series.value = delta_cost.dropna()

        # Calculate the costs and stores the original value
        costs = self.electricity_costs_series.integrate(time_unit="hours")

        # correct the costs
        corrected_costs = costs - stored_energy_diff * np.mean(electricity_price_signal)

        self.costs.value = costs
        self.corrected_costs.value = corrected_costs

    def _calculate_costs_rel(self):
        """Calculate the relative costs of the flexibility event per energy flexibility."""
        if self.energy_flex.value == 0:
            costs_rel = 0
            corrected_costs_rel = 0
        else:
            costs_rel = self.costs.value / self.energy_flex.value
            corrected_costs_rel = self.corrected_costs.value / self.energy_flex.value

        # Set value
        self.costs_rel.value = costs_rel
        self.corrected_costs_rel.value = corrected_costs_rel

    def get_kpi_dict(self, identifier: bool = False) -> dict[str, KPI]:
        """Get the KPIs as a dictionary with names or identifier as keys.

        Args:
            identifier: If True, the keys are the identifiers of the KPIs,
            otherwise the name of the KPI.

        Returns:
            A dictionary mapping desired KPI keys to KPI.

        """
        kpi_dict = {}
        for kpi in vars(self).values():
            if isinstance(kpi, KPI):
                if identifier:
                    kpi_dict[kpi.get_kpi_identifier()] = kpi
                else:
                    kpi_dict[kpi.name] = kpi
        return kpi_dict

    def get_name_dict(self) -> dict[str, str]:
        """Get KPIs mapping.

        Returns:
            Dictionary of the kpis with names as keys and the identifiers as values.

        """
        name_dict = {}
        for name, kpi in self.get_kpi_dict(identifier=False).items():
            name_dict[name] = kpi.get_kpi_identifier()
        return name_dict


class FlexibilityData(pydantic.BaseModel):
    """Class containing the data for the calculation of the flexibility."""

    # Time parameters
    mpc_time_grid: np.ndarray = pydantic.Field(
        default=None,
        description="Time grid of the mpcs",
    )
    flex_offer_time_grid: np.ndarray = pydantic.Field(
        default=None,
        description="Time grid of the flexibility offer",
    )
    switch_time: Optional[float] = pydantic.Field(
        default=None,
        description="Time of the switch between the preparation and the market time",
    )

    # Profiles
    power_profile_base: pd.Series = pydantic.Field(
        default=None,
        description="Base power profile",
    )
    power_profile_flex_neg: pd.Series = pydantic.Field(
        default=None,
        description="Power profile of the negative flexibility",
    )
    power_profile_flex_pos: pd.Series = pydantic.Field(
        default=None,
        description="Power profile of the positive flexibility",
    )
    stored_energy_profile_base: pd.Series = pydantic.Field(
        default=None,
        description="Base profile of the stored electrical energy",
    )
    stored_energy_profile_flex_neg: pd.Series = pydantic.Field(
        default=None,
        description="Profile of the stored electrical energy for negative flexibility",
    )
    stored_energy_profile_flex_pos: pd.Series = pydantic.Field(
        default=None,
        description="Profile of the stored elctrical energy for positive flexibility",
    )
    electricity_price_series: pd.Series = pydantic.Field(
        default=None,
        description="Profile of the electricity price",
    )
    feed_in_price_series: pd.Series = pydantic.Field(
        default=None,
        description="Profile of the electricity feed-in price",
    )

    # KPIs
    kpis_pos: FlexibilityKPIs = pydantic.Field(
        default=FlexibilityKPIs(direction="positive"),
        description="KPIs for positive flexibility",
    )
    kpis_neg: FlexibilityKPIs = pydantic.Field(
        default=FlexibilityKPIs(direction="negative"),
        description="KPIs for negative flexibility",
    )

    class Config:
        """Allow arbitrary (non-Pydantic) types such as pandas.Series or numpy.ndarray
        in model fields without requiring custom validators."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        prep_time: int,
        market_time: int,
        flex_event_duration: int,
        time_step: int,
        prediction_horizon: int,
        **data,
    ):
        super().__init__(**data)
        self.switch_time = prep_time + market_time
        self.flex_offer_time_grid = np.arange(
            self.switch_time, self.switch_time + flex_event_duration + time_step, time_step
        )
        self.mpc_time_grid = np.arange(0, prediction_horizon * time_step + time_step, time_step)
        self._common_time_grid = None  # Initialize common time grid

    def unify_inputs(self, series: pd.Series, mpc=True) -> pd.Series:
        """Format the input of the mpc to unify the data.

        Args:
            series: Input series from a mpc.

        Returns:
            Formatted series.

        """
        if mpc:
            series = series
        else:
            series.index = series.index - series.index[0]

        # Ensure series has values at mpc_time_grid points
        mpc_points_in_series = np.isin(self.mpc_time_grid, series.index)
        if not all(mpc_points_in_series):
            # Create a temp series with all mpc points
            temp_series = pd.Series(index=pd.Index(self.mpc_time_grid), dtype=series.dtype)
            # Merge with original series
            merged_series = pd.concat([series, temp_series])
            # Remove duplicates keeping the original values
            merged_series = merged_series[~merged_series.index.duplicated(keep="first")]
            # Sort by index
            series = merged_series.sort_index()
        # Fill NaNs
        if mpc:
            # only fill NaN if there is NaN except for the first value
            if any(np.isnan(series.loc[1:])):
                series = fill_nans(series=series, method=MEAN)
            # ensure the first value is nan, since it is calculated with the state from the
            # controlled system and thus the same for baseline and shadow mpcs
            series.iloc[0] = np.nan

        if not mpc:
            series = series.ffill()  # price signals are typically steps

        # Check for NaNs
        if any(series.loc[1:].isna()):
            raise ValueError(
                f"The mpc time grid is not compatible with the mpc input "
                f"provided for kpi calculation, "
                f"which leads to NaN values in the series.\n"
                f"MPC time grid:{self.mpc_time_grid}\n"
                f"Series index:{series.index} \n"
                f"Check time steps of the mpcs as well as casadi simulator "
                f"step sizes."
            )
        return series

    def calculate(
        self,
        enable_energy_costs_correction: bool,
        calculate_flex_cost: bool,
        integration_method: INTEGRATION_METHOD,
        collocation_time_grid: list = None,
    ):
        """Calculate the KPIs for the positive and negative flexibility.

        Args:
            enable_energy_costs_correction: whether the energy costs should be corrected
            calculate_flex_cost: whether the cost of the flexibility should be calculated
            integration_method: method used for integration of KPISeries e.g. linear, constant
            collocation_time_grid: Time grid of the mpc output with collocation discretization

        """
        self.kpis_pos.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_shadow=self.power_profile_flex_pos,
            electricity_price_series=self.electricity_price_series,
            feed_in_price_series=self.feed_in_price_series,
            mpc_time_grid=self.mpc_time_grid,
            flex_offer_time_grid=self.flex_offer_time_grid,
            stored_energy_base=self.stored_energy_profile_base,
            stored_energy_shadow=self.stored_energy_profile_flex_pos,
            enable_energy_costs_correction=enable_energy_costs_correction,
            calculate_flex_cost=calculate_flex_cost,
            integration_method=integration_method,
            collocation_time_grid=collocation_time_grid,
        )
        self.kpis_neg.calculate(
            power_profile_base=self.power_profile_base,
            power_profile_shadow=self.power_profile_flex_neg,
            electricity_price_series=self.electricity_price_series,
            feed_in_price_series=self.feed_in_price_series,
            mpc_time_grid=self.mpc_time_grid,
            flex_offer_time_grid=self.flex_offer_time_grid,
            stored_energy_base=self.stored_energy_profile_base,
            stored_energy_shadow=self.stored_energy_profile_flex_neg,
            enable_energy_costs_correction=enable_energy_costs_correction,
            calculate_flex_cost=calculate_flex_cost,
            integration_method=integration_method,
            collocation_time_grid=collocation_time_grid,
        )
        self.reset_time_grid()
        return self.kpis_pos, self.kpis_neg

    def get_kpis(self) -> dict[str, KPI]:
        """Return combined KPIs from positive and negative flexibility scenarios."""
        kpis_dict = self.kpis_pos.get_kpi_dict(identifier=True) | self.kpis_neg.get_kpi_dict(
            identifier=True
        )
        return kpis_dict

    def reset_time_grid(self):
        """
        Reset the common time grid.
        This should be called between different flexibility calculations.
        """
        self._common_time_grid = None

    def update_profile(self, name: str, value: pd.Series, mpc:bool) -> None:
        """Update a specific profile for calculation with a new value."""
        if value is not None: 
            value = self.unify_inputs(series=value, mpc= mpc)
        setattr(self, name, value)

    
