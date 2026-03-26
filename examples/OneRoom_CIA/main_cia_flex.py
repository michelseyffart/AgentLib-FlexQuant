import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from agentlib.utils.multi_agent_system import LocalMASAgency
import numpy as np
import agentlib_mpc.utils.plotting.basic as mpcplot
from agentlib_mpc.utils.analysis import mpc_at_time_step
from agentlib_flexquant.generate_flex_agents import FlexAgentGenerator
import logging
from agentlib_flexquant.utils.interactive import Dashboard

# Set the log-level
logging.basicConfig(level=logging.WARN)
until = 12000

time_of_activation = 9000

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 10}


def run_example(until=until, with_plots=False, with_dashboard=False):
    """
    Runs with the CIA algorithm for solving MINLPs. AgentLib-FlexQuant implements a
    custom optimization backend, that also enables rounding instead of CIA for solving
    these problems, which sometimes shows better performance. To toggle this option set
    use_rounding in the config.

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
    results = []
    mpc_config = "mpc_and_sim/simple_cia_mpc.json"
    sim_config = "mpc_and_sim/simple_cia_sim.json"
    predictor_config = "predictor/predictor_config.json"
    flex_config = "flex_configs/flexibility_agent_config.json"
    agent_configs = [sim_config, predictor_config]

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
        # disturbances
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
        ax1 = axs[0]
        # load
        ax1.set_ylabel("$dot{Q}_{Room}$ in W")
        results["SimAgent"]["room"]["load"].dropna().plot(ax=ax1, drawstyle="steps-post")
        x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(0, 3600 * 6)

        # room temp
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
        ax1 = axs[0]
        # T out
        ax1.set_ylabel("$T_{room}$ in K")
        results["SimAgent"]["room"]["T_upper"].plot(ax=ax1, color="0.5")
        results["SimAgent"]["room"]["T_out"].plot(ax=ax1,
                                                  color=mpcplot.EBCColors.dark_grey)
        mpc_at_time_step(
            data=results["Baseline"]["Baseline"], time_step=time_of_activation,
            variable="T"
        ).plot(ax=ax1, label="base", linestyle="--", color=mpcplot.EBCColors.dark_grey)
        mpc_at_time_step(
            data=results["NegFlexMPC"]["NegFlexMPC"], time_step=time_of_activation,
            variable="T"
        ).plot(ax=ax1, label="neg", linestyle="--", color=mpcplot.EBCColors.red)
        mpc_at_time_step(
            data=results["PosFlexMPC"]["PosFlexMPC"], time_step=time_of_activation,
            variable="T"
        ).plot(ax=ax1, label="pos", linestyle="--", color=mpcplot.EBCColors.blue)

        ax1.legend()
        ax1.vlines(time_of_activation, ymin=0, ymax=500, colors="black")
        ax1.vlines(time_of_activation + 300, ymin=0, ymax=500, colors="black")
        ax1.vlines(time_of_activation + 600, ymin=0, ymax=500, colors="black")
        ax1.vlines(time_of_activation + 3000, ymin=0, ymax=500, colors="black")

        ax1.set_ylim(289, 299)
        x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(0, 3600 * 6)

        # predictions
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
        ax1 = axs[0]
        # P_el
        ax1.set_ylabel("$P_{el}$ in W")
        results["SimAgent"]["room"]["P_el"].plot(ax=ax1,
                                                 color=mpcplot.EBCColors.dark_grey)
        mpc_at_time_step(
            data=results["NegFlexMPC"]["NegFlexMPC"], time_step=time_of_activation,
            variable="P_el"
        ).ffill().plot(
            ax=ax1,
            drawstyle="steps-post",
            label="neg",
            linestyle="--",
            color=mpcplot.EBCColors.red,
        )
        mpc_at_time_step(
            data=results["PosFlexMPC"]["PosFlexMPC"], time_step=time_of_activation,
            variable="P_el"
        ).ffill().plot(
            ax=ax1,
            drawstyle="steps-post",
            label="pos",
            linestyle="--",
            color=mpcplot.EBCColors.blue,
        )
        mpc_at_time_step(
            data=results["Baseline"]["Baseline"], time_step=time_of_activation,
            variable="P_el"
        ).ffill().plot(
            ax=ax1,
            drawstyle="steps-post",
            label="base",
            linestyle="--",
            color=mpcplot.EBCColors.dark_grey,
        )
        ax1.legend()
        ax1.vlines(time_of_activation, ymin=0, ymax=500, colors="black")
        ax1.vlines(time_of_activation + 300, ymin=0, ymax=500, colors="black")
        ax1.vlines(time_of_activation + 600, ymin=0, ymax=500, colors="black")
        ax1.vlines(time_of_activation + 3000, ymin=0, ymax=500, colors="black")

        # flexibility
        # get only the first prediction time of each time step
        ind_res = results["FlexibilityIndicator"]["FlexibilityIndicator"]
        energy_flex_neg = ind_res.xs("negative_energy_flex", axis=1).droplevel(
            1).dropna()
        energy_flex_pos = ind_res.xs("positive_energy_flex", axis=1).droplevel(
            1).dropna()
        fig, axs = mpcplot.make_fig(style=mpcplot.Style(use_tex=False), rows=1)
        ax1 = axs[0]
        ax1.set_ylabel("$epsilon$ in kWh")
        energy_flex_neg.plot(ax=ax1, label="neg", color=mpcplot.EBCColors.red)
        energy_flex_pos.plot(ax=ax1, label="pos", color=mpcplot.EBCColors.blue)
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

        ax1.legend()

        x_ticks = np.arange(0, 3600 * 6 + 1, 3600)
        x_tick_labels = [int(tick / 3600) for tick in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels)
        ax1.set_xlabel("Time in hours")
        for ax in axs:
            mpcplot.make_grid(ax)
            ax.set_xlim(0, 3600 * 6)
        plt.show()

    if with_dashboard:
        Dashboard(
            flex_config="flex_configs/flexibility_agent_config.json",
            simulator_agent_config="mpc_and_sim/simple_cia_sim.json",
            results=results
        ).show()

    return results


if __name__ == "__main__":
    run_example(until, with_plots=True, with_dashboard=False)
