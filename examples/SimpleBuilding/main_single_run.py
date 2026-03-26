import logging
from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator
from agentlib.utils.multi_agent_system import LocalMASAgency

logging.basicConfig(level=logging.WARN)
until = 3600 * 24 

ENV_CONFIG = {"rt": False, "factor": 0.002, "t_sample": 1}
sim_config = "mpc_and_sim/fmu_config.json"
mpc_config = "mpc_and_sim/simple_building.json"
predictor_config = "predictor/predictor_config.json"
flex_config = "flex_configs/flexibility_agent_config.json"

def run_example(until=until):
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

    generator = FlexAgentGenerator(
        flex_config=flex_config, mpc_agent_config=mpc_config
    )

    config_list = generator.generate_flex_agents()
    sim_config_new = generator.adapt_sim_results_path(sim_config, save_name_suffix="_temp")

    agent_configs = [sim_config_new, predictor_config]
    agent_configs.extend(config_list)

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )
    mas.run(until=until)  
    results = mas.get_results(cleanup=False)
    return results


if __name__ == "__main__":
    # Here the simulation is run once, 
    # generated files are stored in --> the current working directory
    # For an example with multiple runs, see: examples\SimpleBuilding\main_multi_run.py
    # For plotting of results generated from this main file, 
    # see: examples\SimpleBuilding\plot_results_single.py
    run_example(until)
