"""
Defines shadow MPC and MINLP-MPC for positive/negative flexibility quantification.
"""
import os
import math
import numpy as np
import pandas as pd
from pydantic import Field
from typing import Dict, Union, Optional
from collections.abc import Iterable
from agentlib.core.datamodels import AgentVariable, Source
from agentlib_mpc.modules.mpc import mpc_full, minlp_mpc
from agentlib_flexquant.utils.data_handling import fill_nans, MEAN
from agentlib_flexquant.data_structures.globals import (full_trajectory_suffix,
                                                        base_vars_to_communicate_suffix)
import agentlib_flexquant.data_structures.globals as glbs
from agentlib_flexquant.optimization_backends.constrained_cia import ConstrainedCasADiCIABackend


class FlexibilityShadowMPCConfig(mpc_full.MPCConfig):

    baseline_input_names: list[str] = Field(default=[])
    custom_input_names: list[Dict] = Field(default=[])
    baseline_state_names: list[str] = Field(default=[])
    full_control_names: list[str] = Field(default=[])


    baseline_agent_id: str = ""

    casadi_sim_time_step: int = Field(
        default=0,
        description="Time step for simulation with Casadi simulator. "
                    "Value is read from FlexQuantConfig",
    )
    power_variable_name: str = Field(
        default=None, description="Name of the power variable in the "
                                  "shadow mpc model."
    )
    storage_variable_name: Optional[str] = Field(
        default=None, description="Name of the storage variable in the "
                                  "shadow mpc model."
    )


class FlexibilityShadowMPC(mpc_full.MPC):
    """Shadow MPC for calculating positive/negative flexibility offers."""

    config: FlexibilityShadowMPCConfig

    def __init__(self, *args, **kwargs):
        # initialize flex_results with None
        self.flex_results = None

        super().__init__(*args, **kwargs)

        # setup look up dict to track incoming inputs and states
        # (maps name as str to actual AgentVariable)
        input_names_list = [var["name"] for var in self.config.custom_input_names]
        self._track_base_comm_vars_dict: Dict[str, Union[AgentVariable, None]] = {}
        for comm_var in self.config.inputs + self.config.states:
            if (comm_var.name in self.config.full_control_names or
                    comm_var.name + base_vars_to_communicate_suffix in
                    self.config.baseline_input_names or
                    comm_var.name + base_vars_to_communicate_suffix in
                    self.config.baseline_state_names or
                    comm_var.name in input_names_list):
                comm_var.value = None
                self._track_base_comm_vars_dict[comm_var.name] = comm_var.copy(deep=True)
        # set up necessary components if simulation is enabled
        if self.config.casadi_sim_time_step > 0:
            # generate a separate simulation model for integration to ensure
            # the model used in MPC optimization remains unaffected
            self.flex_model = type(self.model)(dt=self.config.casadi_sim_time_step)
            # generate the filename for the simulation results
            self.res_file_flex = self.config.optimization_backend["results_file"].replace(
                "_flex", "_sim_flex"
            )
            # clear the casadi simulator result at the first time step if already exists
            try:
                os.remove(self.res_file_flex)
            except FileNotFoundError:
                pass

    def set_output(self, solution):
        """Takes the solution from optimization backend and sends it to AgentVariables."""
        # Output must be defined in the config as "type"="pd.Series"
        if not self.config.set_outputs:
            return
        self.logger.info("Sending optimal output values to data_broker.")
        df = solution.df
        self.sim_flex_model(solution)
        if self.flex_results is not None:
            for output in self.var_ref.outputs:
                if output not in [
                    self.config.power_variable_name,
                    self.config.storage_variable_name,
                ]:
                    series = df.variable[output]
                    self.set(output, series)
            # send the power and storage variable value from simulation results
            upsampled_output_power = self.flex_results[self.config.power_variable_name]
            self.set(self.config.power_variable_name, upsampled_output_power)
            if self.config.storage_variable_name is not None:
                upsampled_output_storage = self.flex_results[self.config.storage_variable_name]
                self.set(self.config.storage_variable_name, upsampled_output_storage.dropna())
        else:
            for output in self.var_ref.outputs:
                series = df.variable[output]
                self.set(output, series)

    def sim_flex_model(self, solution):
        """simulate the flex model over the preditcion horizon and save results"""

        # return if sim_time_step is not a positive integer and system is in provision
        if not (self.config.casadi_sim_time_step > 0 and not self.get(glbs.PROVISION_VAR_NAME).value):
            return

        # read the defined simulation time step
        sim_time_step = self.config.casadi_sim_time_step
        mpc_time_step = self.config.time_step

        # set the horizon length and the number of simulation steps
        total_horizon_time = int(self.config.prediction_horizon * self.config.time_step)
        n_simulation_steps = math.ceil(total_horizon_time / sim_time_step)

        # read the current optimization result
        result_df = solution.df

        # initialize the flex sim results Dataframe
        self._initialize_flex_results(
            n_simulation_steps, total_horizon_time, sim_time_step, result_df
        )

        # Update model parameters and initial states
        self._update_model_parameters()
        self._update_initial_states(result_df)

        # Run simulation
        self._run_simulation(
            n_simulation_steps, sim_time_step, mpc_time_step, result_df, total_horizon_time
        )

        # set index of flex results to the same as mpc result
        store_results_df = self.flex_results.copy(deep=True)
        store_results_df.index = self.flex_results.index.tolist()

        # save results
        if not os.path.exists(self.res_file_flex):
            store_results_df.to_csv(self.res_file_flex)
        else:
            store_results_df.to_csv(self.res_file_flex, mode="a", header=False)

        # set the flex results format same as mpc result while updating Agentvariable
        self.flex_results.index = self.flex_results.index.get_level_values(1)

    def register_callbacks(self):
        for control_var in self.config.controls:
            self.agent.data_broker.register_callback(
                name=control_var.name + full_trajectory_suffix,
                alias=control_var.name + full_trajectory_suffix,
                callback=self.calc_flex_callback,
                source=Source(agent_id=self.config.baseline_agent_id, module_id=None)
            )
        for base_inputs in self.config.baseline_input_names:
            self.agent.data_broker.register_callback(
                name=base_inputs.removesuffix(base_vars_to_communicate_suffix),  # update MPC variable
                alias=base_inputs,
                callback=self.calc_flex_callback,
                source=Source(agent_id=self.config.baseline_agent_id, module_id=None)
            )
        for custom_inputs in self.config.custom_input_names:
            self.agent.data_broker.register_callback(
                name=custom_inputs["name"],
                alias=custom_inputs["alias"],
                callback=self.calc_flex_callback
            )
        for base_states in self.config.baseline_state_names:
            self.agent.data_broker.register_callback(
                name=base_states.removesuffix(base_vars_to_communicate_suffix),  # update MPC variable
                alias=base_states,
                callback=self.calc_flex_callback,
                source=Source(agent_id=self.config.baseline_agent_id, module_id=None)
            )
        super().register_callbacks()

    def calc_flex_callback(self, inp: AgentVariable, name: str):
        """Ensure that all control trajectories and Baseline inputs/states
        have been set before starting the calculation.

        """
        # during provision do not calculate flex
        if self.get(glbs.PROVISION_VAR_NAME).value:
            return

        # do not trigger callback on self set variables
        if self.agent.config.id == inp.source.agent_id:
            return

        # get the value of the input
        vals = inp.value

        if inp.name in self.config.full_control_names:
            if vals.isna().any():
                vals = fill_nans(series=vals, method=MEAN)
        # add time shift env.time to the incoming variable to adapt to mpc output,
        # which starts at t=0
        if vals.index[0] == 0:
            self.logger.info(f"The incoming variable {inp.name} starts with a time "
                             f"index of 0. Adding the current environment time.")
            vals.index += self.env.time

        # update value in the tracking dictionary
        self._track_base_comm_vars_dict[name].value = vals
        # set value
        self.set(name, vals)

        # make sure all necessary inputs are set
        if all(x.value is not None for x in self._track_base_comm_vars_dict.values()):
            self.do_step()
            for _, comm_var in self._track_base_comm_vars_dict.items():
                comm_var.value = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()

    def _initialize_flex_results(
        self, n_simulation_steps, horizon_length, sim_time_step, result_df
    ):
        """Initialize the flex results dataframe with the correct dimension
        and index and fill with existing results from optimization

        """

        # create MultiIndex for collocation points
        index_coll = pd.MultiIndex.from_arrays(
            [[self.env.now] * len(result_df.index), result_df.index],
            names=["time_step", "time"]
            # Match the names with multi_index but note they're reversed
        )
        # create Multiindex for full simulation sample times
        index_full_sample = pd.MultiIndex.from_tuples(
            zip(
                [self.env.now] * (n_simulation_steps + 1),
                range(0, horizon_length + sim_time_step, sim_time_step),
            ),
            names=["time_step", "time"],
        )
        # merge indexes
        new_index = index_coll.union(index_full_sample).sort_values()
        # initialize the flex results with correct dimension
        self.flex_results = pd.DataFrame(np.nan,
                                         index=new_index,
                                         columns=self.var_ref.outputs)

        # Get the optimization outputs and create a series for fixed
        # optimization outputs with the correct MultiIndex format
        opti_outputs = result_df.variable[self.config.power_variable_name]
        fixed_opti_output = pd.Series(
            opti_outputs.values,
            index=index_coll,
        )
        # fill the output value at the time step where it already exists
        # in optimization output
        for idx in fixed_opti_output.index:
            if idx in self.flex_results.index:
                self.flex_results.loc[idx, self.config.power_variable_name] = (
                    fixed_opti_output)[idx]

    def _update_model_parameters(self):
        """update the value of module parameters with value from config,
        since creating a model just reads the value in the model class
        but not the config.

        """

        for par in self.config.parameters:
            self.flex_model.set(par.name, par.value)

    def _update_initial_states(self, result_df):
        """set the initial value of states"""

        # get state values from the mpc optimization result
        state_values = result_df.variable[self.var_ref.states]
        # update state values with last measurement
        for state, value in zip(self.var_ref.states, state_values.iloc[0]):
            self.flex_model.set(state, value)

    def _run_simulation(
        self, n_simulation_steps, sim_time_step, mpc_time_step, result_df, total_horizon_time
    ):
        """simulate with flex model over the prediction horizon

        """

        # get control and input values from the mpc optimization result
        control_values = result_df.variable[self.var_ref.controls].dropna()
        input_values = result_df.parameter[self.var_ref.inputs].dropna()

        # Get the simulation time step index
        sim_time_index = np.arange(0, (n_simulation_steps + 1) * sim_time_step, sim_time_step)

        # Reindex the controls and inputs to sim_time_index
        control_values_full = control_values.copy().reindex(sim_time_index, method="ffill")
        input_values_full = input_values.copy().reindex(sim_time_index, method="nearest")

        for i in range(0, n_simulation_steps):
            current_sim_time = i * sim_time_step

            # Apply control and input values from the appropriate MPC step
            for control, value in zip(
                self.var_ref.controls, control_values_full.loc[current_sim_time]
            ):
                self.flex_model.set(control, value)

            for input_var, value in zip(
                self.var_ref.inputs, input_values_full.loc[current_sim_time]
            ):
                # change the type of iterable input, since casadi model can't deal with iterable
                if issubclass(eval(self.flex_model.get(input_var).type), Iterable):
                    self.flex_model.get(input_var).type = type(value).__name__
                self.flex_model.set(input_var, value)

            # do integration
            # reduce the simulation time step so that the total horizon time will not be exceeded
            if current_sim_time + sim_time_step <= total_horizon_time:
                t_sample = sim_time_step
            else:
                t_sample = total_horizon_time - current_sim_time
            self.flex_model.do_step(t_start=0, t_sample=t_sample)

            # save output
            for output in self.var_ref.outputs:
                self.flex_results.loc[
                    (self.env.now, current_sim_time + t_sample), output
                ] = self.flex_model.get_output(output).value


class FlexibilityShadowMINLPMPCConfig(minlp_mpc.MINLPMPCConfig):

    baseline_input_names: list[str] = Field(default=[])
    custom_input_names: list[Dict] = Field(default=[])
    baseline_state_names: list[str] = Field(default=[])
    full_control_names: list[str] = Field(default=[])

    baseline_agent_id: str = ""

    casadi_sim_time_step: int = Field(
        default=0,
        description="Time step for simulation with Casadi simulator. "
                    "Value is read from FlexQuantConfig",
    )
    power_variable_name: str = Field(
        default=None, description="Name of the power variable in the "
                                  "shadow mpc model."
    )
    storage_variable_name: Optional[str] = Field(
        default=None, description="Name of the storage variable in the "
                                  "shadow mpc model."
    )


class FlexibilityShadowMINLPMPC(minlp_mpc.MINLPMPC):
    """Shadow MINLP-MPC for calculating positive/negatives flexibility offers.

    """

    config: FlexibilityShadowMINLPMPCConfig

    def __init__(self, *args, **kwargs):
        # initialize flex_results with None
        self.flex_results = None

        super().__init__(*args, **kwargs)

        # setup look up dict to track incoming inputs and states
        # (maps name as str to actual AgentVariable)
        input_names_list = [var["name"] for var in self.config.custom_input_names]
        self._track_base_comm_vars_dict: Dict[str, Union[AgentVariable, None]] = {}
        for comm_var in self.config.inputs + self.config.states:
            if (comm_var.name in self.config.full_control_names or
                    comm_var.name + base_vars_to_communicate_suffix in
                    self.config.baseline_input_names or
                    comm_var.name + base_vars_to_communicate_suffix in
                    self.config.baseline_state_names or
                    comm_var.name in input_names_list):
                comm_var.value = None
                self._track_base_comm_vars_dict[comm_var.name] = comm_var.copy(deep=True)
        # set up necessary components if simulation is enabled
        if self.config.casadi_sim_time_step > 0:
            # generate a separate simulation model for integration to ensure
            # the model used in MPC optimization remains unaffected
            self.flex_model = type(self.model)(dt=self.config.casadi_sim_time_step)
            # generate the filename for the simulation results
            self.res_file_flex = self.config.optimization_backend["results_file"].replace(
                "_flex", "_sim_flex"
            )
            # clear the casadi simulator result at the first time step if already exists
            try:
                os.remove(self.res_file_flex)
            except FileNotFoundError:
                pass

    def register_callbacks(self):
        for control_var in self.config.controls + self.config.binary_controls:
            self.agent.data_broker.register_callback(
                name=control_var.name + full_trajectory_suffix,
                alias=control_var.name + full_trajectory_suffix,
                callback=self.calc_flex_callback,
                source=Source(agent_id=self.config.baseline_agent_id, module_id=None)
            )
        for base_inputs in self.config.baseline_input_names:
            self.agent.data_broker.register_callback(
                name=base_inputs.removesuffix(base_vars_to_communicate_suffix),  # update MPC variable
                alias=base_inputs,
                callback=self.calc_flex_callback,
                source=Source(agent_id=self.config.baseline_agent_id, module_id=None)
            )
        for custom_inputs in self.config.custom_input_names:
            self.agent.data_broker.register_callback(
                name=custom_inputs["name"],
                alias=custom_inputs["alias"],
                callback=self.calc_flex_callback
            )
        for base_states in self.config.baseline_state_names:
            self.agent.data_broker.register_callback(
                name=base_states.removesuffix(base_vars_to_communicate_suffix),  # update MPC variable
                alias=base_states,
                callback=self.calc_flex_callback,
                source=Source(agent_id=self.config.baseline_agent_id, module_id=None)
            )

        super().register_callbacks()

    def calc_flex_callback(self, inp: AgentVariable, name: str):
        """Ensure that all control trajectories and Baseline inputs/states
         have been set before starting the calculation.

        """
        # during provision do not calculate flex
        if self.get(glbs.PROVISION_VAR_NAME).value:
            return

        # do not trigger callback on self set variables
        if self.agent.config.id == inp.source.agent_id:
            return

        # get the value of the input
        vals = inp.value

        if inp.name in self.config.full_control_names:
            if vals.isna().any():
                vals = fill_nans(series=vals, method=MEAN)
            # set full controls to custom cia backend to constrain during market time
            if isinstance(self.optimization_backend, ConstrainedCasADiCIABackend):
                self.optimization_backend.config.full_controls_dict[inp.name] = vals
        # add time shift env.now to the mpc prediction index if it starts at t=0
        if vals.index[0] == 0:
            vals.index += self.env.time

        # update value in the tracking dictionary
        self._track_base_comm_vars_dict[name].value = vals
        # set value
        self.set(name, vals)

        # make sure all necessary inputs are set
        if all(x.value is not None for x in self._track_base_comm_vars_dict.values()):
            self.do_step()
            for _, comm_var in self._track_base_comm_vars_dict.items():
                comm_var.value = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()

    def set_output(self, solution):
        """Takes the solution from optimization backend and sends
        it to AgentVariables.

        """
        # Output must be defined in the config as "type"="pd.Series"
        if not self.config.set_outputs:
            return
        self.logger.info("Sending optimal output values to data_broker.")

        # simulate with the casadi simulator
        self.sim_flex_model(solution)

        df = solution.df
        if self.flex_results is not None:
            for output in self.var_ref.outputs:
                if output not in [
                    self.config.power_variable_name,
                    self.config.storage_variable_name,
                ]:
                    series = df.variable[output]
                    self.set(output, series)
            # send the power and storage variable value from simulation results
            upsampled_output_power = self.flex_results[self.config.power_variable_name]
            self.set(self.config.power_variable_name, upsampled_output_power)
            if self.config.storage_variable_name is not None:
                upsampled_output_storage = self.flex_results[self.config.storage_variable_name]
                self.set(self.config.storage_variable_name, upsampled_output_storage.dropna())
        else:
            for output in self.var_ref.outputs:
                series = df.variable[output]
                self.set(output, series)

    def sim_flex_model(self, solution):
        """simulate the flex model over the preditcion horizon and save results

        """

        # return if sim_time_step is not a positive integer and system is in provision
        if not (self.config.casadi_sim_time_step > 0 and not self.get(glbs.PROVISION_VAR_NAME).value):
            return

        # read the defined simulation time step
        sim_time_step = self.config.casadi_sim_time_step
        mpc_time_step = self.config.time_step

        # set the horizon length and the number of simulation steps
        total_horizon_time = int(self.config.prediction_horizon * self.config.time_step)
        n_simulation_steps = math.ceil(total_horizon_time / sim_time_step)

        # read the current optimization result
        result_df = solution.df

        # initialize the flex sim results Dataframe
        self._initialize_flex_results(
            n_simulation_steps, total_horizon_time, sim_time_step, result_df
        )

        # Update model parameters and initial states
        self._update_model_parameters()
        self._update_initial_states(result_df)

        # Run simulation
        self._run_simulation(
            n_simulation_steps, sim_time_step, mpc_time_step, result_df, total_horizon_time
        )

        # set index of flex results to the same as mpc result
        store_results_df = self.flex_results.copy(deep=True)
        store_results_df.index = self.flex_results.index.tolist()

        # save results
        if not os.path.exists(self.res_file_flex):
            store_results_df.to_csv(self.res_file_flex)
        else:
            store_results_df.to_csv(self.res_file_flex, mode="a", header=False)

        # set the flex results format same as mpc result while updating Agentvariable
        self.flex_results.index = self.flex_results.index.get_level_values(1)

    def _initialize_flex_results(
        self, n_simulation_steps, horizon_length, sim_time_step, result_df
    ):
        """Initialize the flex results dataframe with the correct dimension
        and index and fill with existing results from optimization

        """

        # create MultiIndex for collocation points
        index_coll = pd.MultiIndex.from_arrays(
            [[self.env.now] * len(result_df.index), result_df.index],
            names=["time_step", "time"]
            # Match the names with multi_index but note they're reversed
        )
        # create Multiindex for full simulation sample times
        index_full_sample = pd.MultiIndex.from_tuples(
            zip(
                [self.env.now] * (n_simulation_steps + 1),
                range(0, horizon_length + sim_time_step, sim_time_step),
            ),
            names=["time_step", "time"],
        )
        # merge indexes
        new_index = index_coll.union(index_full_sample).sort_values()
        # initialize the flex results with correct dimension
        self.flex_results = pd.DataFrame(np.nan, index=new_index, columns=self.var_ref.outputs)

        # Get the optimization outputs and create a series for fixed optimization outputs with the
        # correct MultiIndex format
        opti_outputs = result_df.variable[self.config.power_variable_name]
        fixed_opti_output = pd.Series(
            opti_outputs.values,
            index=index_coll,
        )
        # fill the output value at the time step where it already exists in optimization output
        for idx in fixed_opti_output.index:
            if idx in self.flex_results.index:
                self.flex_results.loc[idx, self.config.power_variable_name] = fixed_opti_output[idx]

    def _update_model_parameters(self):
        """update the value of module parameters with value from config,
        since creating a model just reads the value in the model class but not the config
        """

        for par in self.config.parameters:
            self.flex_model.set(par.name, par.value)

    def _update_initial_states(self, result_df):
        """set the initial value of states"""

        # get state values from the mpc optimization result
        state_values = result_df.variable[self.var_ref.states]
        # update state values with last measurement
        for state, value in zip(self.var_ref.states, state_values.iloc[0]):
            self.flex_model.set(state, value)

    def _run_simulation(
        self, n_simulation_steps, sim_time_step, mpc_time_step, result_df, total_horizon_time
    ):
        """simulate with flex model over the prediction horizon"""

        # get control and input values from the mpc optimization result
        control_values = result_df.variable[
            [*self.var_ref.controls, *self.var_ref.binary_controls]
        ].dropna()
        input_values = result_df.parameter[self.var_ref.inputs].dropna()

        # Get the simulation time step index
        sim_time_index = np.arange(0, (n_simulation_steps + 1) * sim_time_step, sim_time_step)

        # Reindex the controls and inputs to sim_time_index
        control_values_full = control_values.copy().reindex(sim_time_index, method="ffill")
        input_values_full = input_values.copy().reindex(sim_time_index, method="nearest")

        for i in range(0, n_simulation_steps):
            current_sim_time = i * sim_time_step

            # Apply control and input values from the appropriate MPC step
            for control, value in zip(
                self.var_ref.controls,
                control_values_full.loc[current_sim_time, self.var_ref.controls],
            ):
                self.flex_model.set(control, value)

            for binary_control, value in zip(
                self.var_ref.binary_controls,
                control_values_full.loc[current_sim_time, self.var_ref.binary_controls],
            ):
                self.flex_model.set(binary_control, value)

            for input_var, value in zip(
                self.var_ref.inputs, input_values_full.loc[current_sim_time]
            ):
                # change the type of iterable input, since casadi model can't deal with iterable
                if issubclass(eval(self.flex_model.get(input_var).type), Iterable):
                    self.flex_model.get(input_var).type = type(value).__name__
                self.flex_model.set(input_var, value)

            # do integration
            # reduce the simulation time step so that the total horizon time will not be exceeded
            if current_sim_time + sim_time_step <= total_horizon_time:
                t_sample = sim_time_step
            else:
                t_sample = total_horizon_time - current_sim_time
            self.flex_model.do_step(t_start=0, t_sample=t_sample)

            # save output
            for output in self.var_ref.outputs:
                self.flex_results.loc[
                    (self.env.now, current_sim_time + t_sample), output
                ] = self.flex_model.get_output(output).value
