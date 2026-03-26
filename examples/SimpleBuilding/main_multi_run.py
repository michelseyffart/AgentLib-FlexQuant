import os
import tempfile
import json
import logging
from copy import deepcopy

from agentlib.utils.multi_agent_system import LocalMASAgency

from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator
from agentlib_flexquant.utils.interactive import Dashboard, CustomBound


logging.basicConfig(level=logging.WARN)
until = 3600 * 24 

ENV_CONFIG = {"rt": False, "factor": 0.002, "t_sample": 1} 


def update_configs(flex_config, flex_event_duration):
    """Loads and modifies a flex config, returns path to updated temporary config file."""
    
    # load config and create a copy
    with open(flex_config, "r") as f:
        flex_config = json.load(f)
    flex_config_copy = deepcopy(flex_config)
    
    # modify parameter and set custom base path for generated flex files
    # base path below can be replaced with your own base path, 
    # e.g. C:/Users/abc-xyz/flex_test/run_1
    flex_config_copy["flex_event_duration"] = flex_event_duration
    flex_config_copy["flex_base_directory_path"] = (
        f"generated_files/run_with_flex_event_duration_{flex_event_duration}"
    )
    
    # dump temp modified flex config for further processing
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
    json.dump(flex_config_copy, temp_file, indent=2)
    temp_file_path = temp_file.name
    temp_file.close()

    return temp_file_path


def run_example(flex_event_duration, until=until):
    """Runs MAS simulation with specified flex event duration.
    
    mpc_config: 
        Sets inputs, outputs, states, and parameters for the MPC agent. 
        It points to the path of the MPC problem definition file (simple_building.py) and defines the MPC parameters.
    sim_config: 
        Sets inputs, outputs, and states for the simulation agent. 
        It points to the path of the FMU file and defines the simulation parameters.
    predictor_config:
        Sets parameters for the predictor agent and points to the path of the predictor formulation file (predictor.py).
    flex_config:
        Sets various options for the flexibility quantification framework: 
        - characteristic times for the indicator module (e.g. market time, preparation time, flex event duration)
        - options for the cost calculation
            - whether to use a constant electricity price or to input a time series sent by the predictor agent
            - whether to use a constant feed-in tariff or to input a time series sent by the predictor agent
                - if no feed-in is required (e.g. for a house without electricity generation), use a constant feed-in tariff with value 0
        - option to correct the cost for stored energy at the end of the prediction horizon 
        - option to include a market config (points to a market config file) 
        - options for the flexibility agent generator: 
            - power variable of the baseline agent 
            - cost functions of PF-MPC and NF-MPC agents, including custom parameters and variables for the shadow MPCs 
        - general options such as results paths 

    """
    
    mpc_config = "mpc_and_sim/simple_building.json"
    sim_config = "mpc_and_sim/fmu_config.json" 
    predictor_config = "predictor/predictor_config.json"
    flex_config = "flex_configs/flexibility_agent_config.json"

    updated_flex_config_path = update_configs(
        flex_config, flex_event_duration)

    generator = FlexAgentGenerator(
        flex_config=updated_flex_config_path, 
        mpc_agent_config=mpc_config
    )

    config_list = generator.generate_flex_agents()
    sim_config = generator.adapt_sim_results_path(sim_config)

    agent_configs = [sim_config, predictor_config, *config_list]

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )
    mas.run(until=until)

    if os.path.exists(updated_flex_config_path):
        os.remove(updated_flex_config_path)


if __name__ == "__main__":
    flex_event_durations = [7200, 8100, 6300]
    for flex_event_duration in flex_event_durations:
        # Here the simulation is run multiple times,
        # generated files are stored in --> custom base paths
        # For an example with single run, see: examples\SimpleBuilding\main_single_run.py
        # For plotting of results generated from this main file, 
        # see: examples\SimpleBuilding\plot_results_mult.py
        run_example(flex_event_duration, until)
