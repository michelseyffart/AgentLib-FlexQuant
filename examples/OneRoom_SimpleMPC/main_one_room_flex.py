import logging
from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator
from agentlib_flexquant.utils.interactive import Dashboard, CustomBound
from agentlib.utils.multi_agent_system import LocalMASAgency
from plot_results import plot_results

# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 21600

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}


def run_example(until=until, with_plots=False, with_dashboard=False):
    """ Example with market usage

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

    Change flex_power_feedback_method in MarketSpecifications to deal with systems with
    fast or slow inertia.
    See difference in results: when using collocation, power during flex event does not
    fully follow offer. When using constant it does. Reason is the fast inertia of the
    system.

    This examples also introduces the flex_cost_function_appendix field, which allows
    for custom cost function terms. These terms are added to the standard objective
    of the corresponding shadow MPC (the filed does not exist for the Baseline, as
    here you would change the base MPC used for creating the shadow MPCs). You can
    change the field in the flexibility_agent_config to see its impact.

    The time delay in the prediction plots (power profiles do not follow predictions)
    is caused by the collocation points and is only a visual effect.

    """

    mpc_config = "mpc_and_sim/simple_model.json"
    sim_config = "mpc_and_sim/simple_sim.json"
    flex_config = "flex_configs/flexibility_agent_config.json"
    agent_configs = [sim_config]

    config_list = FlexAgentGenerator(
        flex_config=flex_config, mpc_agent_config=mpc_config
    ).generate_flex_agents()
    agent_configs.extend(config_list)

    mas = LocalMASAgency(
        agent_configs=agent_configs, env=ENV_CONFIG, variable_logging=False
    )

    mas.run(until=until)
    results = mas.get_results(cleanup=False)

    if with_plots:
        # Alternative plots using matplotlib
        plot_results(results_data=results)

    if with_dashboard:
        Dashboard(
            flex_config="flex_configs/flexibility_agent_config.json",
            simulator_agent_config="mpc_and_sim/simple_sim.json",
            results=results
        ).show(
            custom_bounds=CustomBound(
                for_variable="T",
                lb_name="T_lower",
                ub_name="T_upper"
            )
        )
    return results


if __name__ == "__main__":
    run_example(until, with_plots=False, with_dashboard=True)