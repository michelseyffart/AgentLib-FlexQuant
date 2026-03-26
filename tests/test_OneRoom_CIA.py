import pytest
import pandas as pd
import os
import sys
from pathlib import Path
import importlib.util
import json
from util import module_cleanup, round_floats_in_structure

# Add the project root to the Python path to allow for absolute imports
# This helps in locating the agentlib_flexquant package if needed
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


def create_dataframe_summary(df: pd.DataFrame, precision: int = 6) -> dict:
    """Create a robust, compact summary of a DataFrame for snapshotting.

    This summary is designed to be insensitive to minor floating-point differences
    while being highly sensitive to meaningful data changes.

    Args:
        df: The pandas DataFrame to summarize.
        precision: The number of decimal places to round float values to.

    Returns:
        A dictionary containing the summary.

    """
    if df is None or df.empty:
        return {"error": "DataFrame is empty or None"}

    # Get descriptive statistics and round them to handle float precision issues
    summary_stats = df.describe().round(precision)

    # Convert the stats DataFrame to a dictionary. This may have tuple keys.
    stats_dict_raw = summary_stats.to_dict()

    # Create a new dictionary, converting any tuple keys into strings.
    # e.g., ('lower', 'P_el') becomes 'lower.P_el'
    stats_dict_clean = {
        ".".join(map(str, k)) if isinstance(k, tuple) else str(k): v
        for k, v in stats_dict_raw.items()
    }

    # Create the final summary object
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "index_start": str(tuple(float(x) for x in df.index.min())),
        "index_end": str(tuple(float(x) for x in df.index.max())),
        "statistics": stats_dict_clean,
        "head_5_rows": df.head(5).round(precision).to_dict(orient='split'),
        "tail_5_rows": df.tail(5).round(precision).to_dict(orient='split'),
    }
    return summary


def assert_frame_matches_summary_snapshot(snapshot, df: pd.DataFrame,
                                          snapshot_name: str):
    """Assert that a DataFrame's summary matches a stored snapshot.

    This function creates a summary of the dataframe and uses pytest-snapshot
    to compare it against a stored version.

    """
    # Create a summary of the dataframe
    summary = create_dataframe_summary(df)

    # Round all numbers in the summary to handle cross-platform differences
    rounded_summary = round_floats_in_structure(summary, precision=1)

    # Convert the summary dictionary to a formatted JSON string
    summary_json = json.dumps(rounded_summary, indent=2, sort_keys=True)

    # Use snapshot.assert_match on the small, stable JSON string
    snapshot.assert_match(summary_json, snapshot_name)

def run_example_from_path(example_path: Path):
    """Dynamically import and run the 'run_example' function from a script
    in the specified directory.

    This function robustly handles changing the working directory AND the
    Python import path, ensuring the script can find both its local files
    and its local modules.

    """
    run_script_path = example_path / 'main_cia_flex.py'
    if not run_script_path.is_file():
        raise FileNotFoundError(
            f"Could not find the run script at {run_script_path}. "
            "Please ensure it is named 'run.py' or adjust the test code."
        )

    # --- SETUP: Store original paths before changing them ---
    original_cwd = Path.cwd()
    original_sys_path = sys.path[:]  # Create a copy of the sys.path list

    module_name = f"agentlib_flexquant.tests.examples.{example_path.name}"

    try:
        # --- STEP 1: Change CWD for file access (e.g., config.json) ---
        os.chdir(example_path)

        # --- STEP 2: Add example dir to sys.path for module imports ---
        sys.path.insert(0, str(example_path))

        # Dynamically import the run_example function from the script
        spec = importlib.util.spec_from_file_location(module_name, run_script_path)
        run_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = run_module
        spec.loader.exec_module(run_module)

        if not hasattr(run_module, 'run_example'):
            raise AttributeError(
                "The 'run.py' script must contain a 'run_example' function.")

        # Execute the function and get the results
        results = run_module.run_example(until=3600, with_plots=True,
                                         with_dashboard=False)
        return results

    finally:
        # --- TEARDOWN: Always restore original paths to avoid side-effects ---
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path  # Restore the original sys.path


def test_oneroom_cia(snapshot, module_cleanup):
    """Unit test for the OneRoom_CIA example using snapshot testing.

    This test runs the example via its own run script and compares the
    full resulting dataframes against stored snapshots.

    """
    # Define the path to the example directory
    example_path = root_path / 'examples' / 'OneRoom_CIA'

    # Run the example and get the results object
    res = run_example_from_path(example_path)

    # Extract the full resulting dataframes as requested
    df_neg_flex_res = res["NegFlexMPC"]["NegFlexMPC"]
    df_pos_flex_res = res["PosFlexMPC"]["PosFlexMPC"]
    df_baseline_res = res["Baseline"]["Baseline"]
    df_indicator_res = res["FlexibilityIndicator"]["FlexibilityIndicator"]

    # Assert that a summary of each result DataFrame matches its snapshot
    assert_frame_matches_summary_snapshot(
        snapshot,
        df_neg_flex_res,
        'oneroom_cia_neg_flex_summary.json'
    )
    assert_frame_matches_summary_snapshot(
        snapshot,
        df_pos_flex_res,
        'oneroom_cia_pos_flex_summary.json'
    )
    assert_frame_matches_summary_snapshot(
        snapshot,
        df_baseline_res,
        'oneroom_cia_baseline_summary.json'
    )
    assert_frame_matches_summary_snapshot(
        snapshot,
        df_indicator_res,
        'oneroom_cia_indicator_summary.json'
    )