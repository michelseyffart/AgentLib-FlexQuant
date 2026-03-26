import pydantic
import numpy as np
from agentlib.core.errors import OptionalDependencyError
from agentlib_mpc.optimization_backends.casadi_.minlp_cia import CasADiCIABackend
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import CasadiBackendConfig
from agentlib_mpc.data_structures.mpc_datamodels import MINLPVariableReference, MPCVariable
from agentlib_flexquant.data_structures.globals import full_trajectory_suffix
from agentlib_mpc.optimization_backends.casadi_.core.discretization import Results

try:
    import pycombina
except ImportError:
    raise OptionalDependencyError(
        used_object="Pycombina",
        dependency_install=".\ after cloning pycombina. Instructions: "
        "https://pycombina.readthedocs.io/en/latest/install.html#",
    )


class ConstrainedCIABackendConfig(CasadiBackendConfig):
    market_time: int = pydantic.Field(
        default=900,
        ge=0,
        unit="s",
        description="Time for market interaction",
    )
    use_rounding: bool = pydantic.Field(
        default=False,
        description="If True, CIA is skipped and plain rounding is used.",
    )
    full_controls_dict: dict = pydantic.Field(
        default={},
        description="Holds a key value pair for each full control of the Baseline",
    )

    class Config:
        # Explicitly set this to allow additional fields in the derived class
        extra = "forbid"


class ConstrainedCasADiCIABackend(CasADiCIABackend):
    var_ref: MINLPVariableReference
    config_type = ConstrainedCIABackendConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, now: float, current_vars: dict[str, MPCVariable]) -> Results:
        # collect and format inputs
        mpc_inputs = self._get_current_mpc_inputs(agent_variables=current_vars, now=now)

        # solve NLP with relaxed binaries
        relaxed_results = self.discretization.solve(mpc_inputs)

        if self.config.use_rounding:
            b_rel = [relaxed_results[var] for var in self.var_ref.binary_controls]
            b_rel_np = np.transpose(np.vstack(b_rel))

            # List to collect all rounded/overwritten binary arrays
            binary_arrays = []

            # constrain shadow MPCs to values of baseline for time<market_time
            for bin_con, bin_rel in zip(self.var_ref.binary_controls, b_rel_np):
                cons = self.get_baseline_binary_solution(bin_con)
                # round binaries
                binary_array = np.round(bin_rel)
                if cons is not None:
                    # Determine how many sample times are before the market time
                    sample_time = self.config.discretization_options.time_step
                    market_time = self.config.market_time
                    num_samples_before_market = int(market_time / sample_time)
                    # Overwrite the market time entries with baseline values
                    for i in range(num_samples_before_market):
                        time_point = i * sample_time
                        if time_point in cons.index:
                            binary_array[i] = cons.loc[time_point]
                binary_arrays.append(binary_array)
            # Stack all binary arrays back together (transpose to match expected shape)
            binary_array = np.transpose(np.vstack(binary_arrays))

        else:
            relaxed_binary_array = self.make_binary_array(full_results=relaxed_results)
            binary_array = self.do_pycombina(b_rel=relaxed_binary_array)

        mpc_inputs_new = self.constrain_binary_inputs(
            mpc_inputs_old=mpc_inputs,
            binary_array=binary_array,
        )
        # solve NLP with fixed binaries
        full_results_final = self.discretization.solve(mpc_inputs_new)

        self.save_rel_result_df(relaxed_results, now=now)
        self.save_result_df(full_results_final, now=now)

        return full_results_final

    def do_pycombina(self, b_rel: np.array) -> np.array:

        grid = self.discretization.grid(self.system.binary_controls).copy()
        grid.append(grid[-1] + self.config.discretization_options.time_step)

        binapprox = pycombina.BinApprox(
            t=grid,
            b_rel=b_rel,
        )

        # constrain shadow MPCs to values of baseline for time<market_time
        for bin_con in self.var_ref.binary_controls:
            cons = self.get_baseline_binary_solution(bin_con)
            if cons is not None:
                last_idx = 0
                for idx, value in cons.items():
                    # constrain every timestep before market_time
                    # with values of baseline
                    binapprox.set_valid_controls_for_interval(
                        (last_idx, idx), [value, 1 - value]
                    )
                    last_idx = idx

        bnb = pycombina.CombinaBnB(binapprox)
        bnb.solve(
            use_warm_start=False,
            max_cpu_time=15,
            verbosity=0,
        )
        b_bin = binapprox.b_bin

        # if there is only one mode, we created a dummy mode which we remove now
        if len(self.var_ref.binary_controls) == 1:
            b_bin = b_bin[0, :].reshape(1, -1)

        return b_bin

    def get_baseline_binary_solution(self, bin_con):
        # check for baseline or shadow MPC
        if not self.config.full_controls_dict:
            # if baseline, return
            return None
        name = bin_con + full_trajectory_suffix
        # if shadow MPC, get current value send by baseline
        if self.config.full_controls_dict[name] is not None:
            cons = self.config.full_controls_dict[name]
            # the index of constraints starts at the absolute current environment
            # time, while the market time is relative time on mpc horizon
            cons.index -= cons.index[0]
            # get the constraints in the market time
            cons = cons[cons.index <= self.config.market_time]
            return cons