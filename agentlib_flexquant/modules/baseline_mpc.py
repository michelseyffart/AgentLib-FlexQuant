"""
Defines MPC and MINLP-MPC for baseline flexibility quantification.
"""
import os
import math
import numpy as np
import pandas as pd
from typing import Optional, Dict
from pydantic import Field
from collections.abc import Iterable
import agentlib_flexquant.data_structures.globals as glbs
from agentlib import AgentVariable
from agentlib_mpc.modules.mpc import mpc_full, minlp_mpc
from agentlib_mpc.data_structures.mpc_datamodels import Results, InitStatus
from agentlib_flexquant.data_structures.globals import (full_trajectory_suffix,
                                                        base_vars_to_communicate_suffix)


class FlexibilityBaselineMPCConfig(mpc_full.MPCConfig):
    # define an AgentVariable list for the full control trajectory
    full_controls: list[AgentVariable] = Field(default=[])

    # define an AgentVariable list for the variables to communicate to the shadow MPCs
    vars_to_communicate: list[AgentVariable] = Field(default=[])

    casadi_sim_time_step: int = Field(
        default=0,
        description="Time step for simulation with Casadi"
        "simulator. Value is read from "
        "FlexQuantConfig",
    )
    power_variable_name: str = Field(
        default=None, description="Name of the power variable in the "
                                  "baseline mpc model."
    )
    storage_variable_name: Optional[str] = Field(
        default=None, description="Name of the storage variable in the "
                                  "baseline mpc model."
    )


class FlexibilityBaselineMPC(mpc_full.MPC):
    """MPC for baseline flexibility quantification."""

    config: FlexibilityBaselineMPCConfig

    def __init__(self, config, agent):
        super().__init__(config, agent)
        # initialize a control mapping dictionary which maps the names of the
        # incoming AgentVariables (with suffix) to the names without suffix
        self._controls_name_mapping: Dict[str, str] = {}
        self._vars_to_com_name_mapping: Dict[str, str] = {}

        for full_control in self.config.full_controls:
            # fill the mapping dictionary
            self._controls_name_mapping[full_control.name] = full_control.name.replace(
                full_trajectory_suffix, ""
            )

        for vars_to_com in self.config.vars_to_communicate:
            # fill the mapping dictionary
            self._vars_to_com_name_mapping[vars_to_com.name] = vars_to_com.name.replace(
                base_vars_to_communicate_suffix, ""
            )

        # initialize flex_results with None
        self.flex_results = None
        # set up necessary components if simulation is enabled
        if self.config.casadi_sim_time_step > 0:
            # generate a separate flex_model for integration to ensure
            # the model used in MPC optimization remains unaffected
            self.flex_model = type(self.model)(dt=self.config.casadi_sim_time_step)
            # generate the filename for the simulation results
            self.res_file_flex = self.config.optimization_backend["results_file"].replace(
                "_base", "_sim_base"
            )
            # clear the casadi simulator result at the first time step if already exists
            try:
                os.remove(self.res_file_flex)
            except FileNotFoundError:
                pass

    def do_step(self):
        """
        Performs an MPC step.
        """
        if not self.init_status == InitStatus.ready:
            self.logger.warning("Skipping step, optimization_backend is not ready.")
            return

        self.pre_computation_hook()

        # get new values from data_broker
        updated_vars = self.collect_variables_for_optimization()

        # solve optimization problem with up-to-date values from data_broker
        result = self.optimization_backend.solve(self.env.time, updated_vars)

        # Set variables in data_broker
        self.set_actuation(result)
        # Set variables, so that shadow MPCs are initialized with the
        # same values as the Baseline
        self.set_vars_for_shadow(result)
        self.set_output(result)
        self._remove_old_values_from_history()

    def pre_computation_hook(self):
        """Calculate relative start and end times for flexibility provision.

        When in provision mode, computes the relative timing for flexibility
        events based on the external power profile timestamps and current
        environment time.
        """
        if self.get(glbs.PROVISION_VAR_NAME).value:
            self.set(glbs.RELATIVE_EVENT_START_TIME_VAR_NAME,
                     self.get(glbs.ACCEPTED_POWER_VAR_NAME).value.index[0] -
                     self.env.time)
            # the provision profile gives a value for the start of a time step.
            # For the end of the flex interval add time step!
            self.set(glbs.RELATIVE_EVENT_END_TIME_VAR_NAME,
                     self.get(glbs.ACCEPTED_POWER_VAR_NAME).value.index[-1] -
                     self.env.time)

    def set_vars_for_shadow(self, solution):
        """Sets the variables of the Baseline MPC needed by the shadow MPCs
        with a predefined suffix.
        This essentially sends the same inputs and states the Baseline used
        for optimization to the Shadow MPC, ensuring synchronisation.
        """
        for vars_to_com in self.config.vars_to_communicate:
            vars_name = self._vars_to_com_name_mapping[vars_to_com.name]
            if vars_name in solution.df.variable:
                vars_value = solution.df.variable[vars_name]
            else:  # parameter
                vars_value = solution.df.parameter[vars_name]
            self.set(vars_to_com.name, vars_value)

    def set_actuation(self, solution: Results):
        super().set_actuation(solution)
        for full_control in self.config.full_controls:
            # get the corresponding control name
            control = self._controls_name_mapping[full_control.name]
            # set value to full_control
            self.set(full_control.name, solution.df.variable[control].ffill())

    def set_output(self, solution):
        """Takes the solution from optimization backend and sends it
        to AgentVariables.

        """
        # Output must be defined in the config as "type"="pd.Series"
        if not self.config.set_outputs:
            return
        self.logger.info("Sending optimal output values to data_broker.")

        # simulate with the casadi simulator
        self.sim_flex_model(solution)

        # extract solution DataFrama
        df = solution.df

        # send the outputs
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
                upsampled_output_storage = (
                    self.flex_results)[self.config.storage_variable_name]
                self.set(self.config.storage_variable_name,
                         upsampled_output_storage.dropna())
        else:
            for output in self.var_ref.outputs:
                series = df.variable[output]
                self.set(output, series)

    def sim_flex_model(self, solution):
        """simulate the flex model over the preditcion horizon and save results

        """

        # return if sim_time_step is not a positive integer and system is in provision
        if not (self.config.casadi_sim_time_step > 0 and not
                self.get(glbs.PROVISION_VAR_NAME).value):
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
            n_simulation_steps, sim_time_step, mpc_time_step, result_df,
            total_horizon_time
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
        """Initialize the flex results dataframe with the correct dimension and index
        and fill with existing results from optimization

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
        self.flex_results = pd.DataFrame(np.nan, index=new_index,
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
        since creating a model just reads the value in the model class but
        not the config

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
        self, n_simulation_steps, sim_time_step, mpc_time_step, result_df,
            total_horizon_time
    ):
        """simulate with flex model over the prediction horizon

        """

        # get control and input values from the mpc optimization result
        control_values = result_df.variable[self.var_ref.controls].dropna()
        input_values = result_df.parameter[self.var_ref.inputs].dropna()

        # Get the simulation time step index
        sim_time_index = np.arange(0, (n_simulation_steps + 1) * sim_time_step,
                                   sim_time_step)

        # Reindex the controls and inputs to sim_time_index
        control_values_full = control_values.copy().reindex(sim_time_index,
                                                            method="ffill")
        input_values_full = input_values.copy().reindex(sim_time_index,
                                                        method="nearest")

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
                # change the type of iterable input, since casadi model
                # can't deal with iterable
                if issubclass(eval(self.flex_model.get(input_var).type), Iterable):
                    self.flex_model.get(input_var).type = type(value).__name__
                self.flex_model.set(input_var, value)

            # do integration
            # reduce the simulation time step so that the total horizon time
            # will not be exceeded
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


class FlexibilityBaselineMINLPMPCConfig(minlp_mpc.MINLPMPCConfig):
    # define an AgentVariable list for the full control trajectory
    full_controls: list[AgentVariable] = Field(default=[])

    # define an AgentVariable list for the variables to communicate to the shadow MPCs
    vars_to_communicate: list[AgentVariable] = Field(default=[])

    casadi_sim_time_step: int = Field(
        default=0,
        description="Time step for simulation with Casadi simulator. "
                    "Value is read from  FlexQuantConfig",
    )
    power_variable_name: str = Field(
        default=None, description="Name of the power variable in the "
                                  "baseline mpc model."
    )
    storage_variable_name: Optional[str] = Field(
        default=None, description="Name of the storage variable in the "
                                  "baseline mpc model."
    )


class FlexibilityBaselineMINLPMPC(minlp_mpc.MINLPMPC):
    """MINLP-MPC for baseline flexibility quantification with mixed-integer
    optimization.

    """

    config: FlexibilityBaselineMINLPMPCConfig

    def __init__(self, config, agent):
        super().__init__(config, agent)
        # initialize a control mapping dictionary which maps the names of the
        # incoming AgentVariables (with suffix) to the names without suffix
        self._controls_name_mapping: Dict[str, str] = {}
        self._vars_to_com_name_mapping: Dict[str, str] = {}

        for full_control in self.config.full_controls:
            # fill the mapping dictionary
            self._controls_name_mapping[full_control.name] = full_control.name.replace(
                full_trajectory_suffix, ""
            )

        for vars_to_com in self.config.vars_to_communicate:
            # fill the mapping dictionary
            self._vars_to_com_name_mapping[vars_to_com.name] = vars_to_com.name.replace(
                base_vars_to_communicate_suffix, ""
            )

        # initialize flex_results with None
        self.flex_results = None
        # set up necessary components if simulation is enabled
        if self.config.casadi_sim_time_step > 0:
            # generate a separate flex_model for integration to ensure
            # the model used in MPC optimization remains unaffected
            self.flex_model = type(self.model)(dt=self.config.casadi_sim_time_step)
            # generate the filename for the simulation results
            self.res_file_flex = self.config.optimization_backend[
                "results_file"].replace("_base", "_sim_base")
            # clear the casadi simulator result at the first time step if already exists
            try:
                os.remove(self.res_file_flex)
            except FileNotFoundError:
                pass

    def do_step(self):
        """
        Performs an MPC step.
        """
        if not self.init_status == InitStatus.ready:
            self.logger.warning("Skipping step, optimization_backend is not ready.")
            return

        self.pre_computation_hook()

        # get new values from data_broker
        updated_vars = self.collect_variables_for_optimization()

        # solve optimization problem with up-to-date values from data_broker
        result = self.optimization_backend.solve(self.env.time, updated_vars)

        # Set variables in data_broker
        self.set_actuation(result)
        # Set variables, so that shadow MPCs are initialized
        # with the same values as the Baseline
        self.set_vars_for_shadow(result)
        self.set_output(result)

    def pre_computation_hook(self):
        """Calculate relative start and end times for flexibility provision.

        When in provision mode, computes the relative timing for flexibility
        events based on the external power profile timestamps and current
        environment time.
        """
        if self.get(glbs.PROVISION_VAR_NAME).value:
            timestep = (self.get(glbs.ACCEPTED_POWER_VAR_NAME).value.index[1] -
                        self.get(glbs.ACCEPTED_POWER_VAR_NAME).value.index[0])
            self.set(glbs.RELATIVE_EVENT_START_TIME_VAR_NAME,
                     self.get(glbs.ACCEPTED_POWER_VAR_NAME).value.index[0] -
                     self.env.time)
            # the provision profile gives a value for the start of a time step.
            # For the end of the flex interval add time step!
            self.set(glbs.RELATIVE_EVENT_END_TIME_VAR_NAME,
                     self.get(glbs.ACCEPTED_POWER_VAR_NAME).value.index[-1] -
                     self.env.time + timestep,)

    def set_vars_for_shadow(self, solution):
        """Sets the variables of the Baseline MPC needed by the shadow MPCs
        with a predefined suffix.
        This essentially sends the same inputs and states the Baseline used
        for optimization to the Shadow MPC, ensuring synchronisation.
        """
        for vars_to_com in self.config.vars_to_communicate:
            vars_name = self._vars_to_com_name_mapping[vars_to_com.name]
            if vars_name in solution.df.variable:
                vars_value = solution.df.variable[vars_name]
            else:  # parameter
                vars_value = solution.df.parameter[vars_name]
            self.set(vars_to_com.name, vars_value)

    def set_actuation(self, solution: Results):
        super().set_actuation(solution)
        for full_control in self.config.full_controls:
            # get the corresponding control name
            control = self._controls_name_mapping[full_control.name]
            # set value to full_control
            self.set(full_control.name, solution.df.variable[control].ffill())

    def set_output(self, solution):
        """Takes the solution from optimization backend and sends it
        to AgentVariables.

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
            upsampled_output_power = (
                self.flex_results)[self.config.power_variable_name]
            self.set(self.config.power_variable_name, upsampled_output_power)
            if self.config.storage_variable_name is not None:
                upsampled_output_storage = (
                    self.flex_results)[self.config.storage_variable_name]
                self.set(self.config.storage_variable_name,
                         upsampled_output_storage.dropna())
        else:
            for output in self.var_ref.outputs:
                series = df.variable[output]
                self.set(output, series)

    def sim_flex_model(self, solution):
        """simulate the flex model over the preditcion horizon and save results

        """

        # return if sim_time_step is not a positive integer and system is in provision
        if not (self.config.casadi_sim_time_step > 0 and not
        self.get(glbs.PROVISION_VAR_NAME).value):
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
            n_simulation_steps, sim_time_step, mpc_time_step, result_df,
            total_horizon_time
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
        self.flex_results = pd.DataFrame(np.nan, index=new_index,
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
        since creating a model just reads the value in the model class but
        not the config

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
        self, n_simulation_steps, sim_time_step, mpc_time_step, result_df,
            total_horizon_time
    ):
        """simulate with flex model over the prediction horizon

        """

        # get control and input values from the mpc optimization result
        control_values = result_df.variable[
            [*self.var_ref.controls, *self.var_ref.binary_controls]
        ].dropna()
        input_values = result_df.parameter[self.var_ref.inputs].dropna()

        # Get the simulation time step index
        sim_time_index = np.arange(0, (n_simulation_steps + 1) * sim_time_step,
                                   sim_time_step)

        # Reindex the controls and inputs to sim_time_index
        control_values_full = control_values.copy().reindex(sim_time_index,
                                                            method="ffill")
        input_values_full = input_values.copy().reindex(sim_time_index,
                                                        method="nearest")

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
                # change the type of iterable input, since casadi model can't
                # deal with iterable
                if issubclass(eval(self.flex_model.get(input_var).type), Iterable):
                    self.flex_model.get(input_var).type = type(value).__name__
                self.flex_model.set(input_var, value)

            # do integration
            # reduce the simulation time step so that the total horizon time
            # will not be exceeded
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
