from pathlib import Path
import pandas as pd
import numpy as np

import chama


# ==============================================================================
# 0. PARALLEL WORKER
# ==============================================================================
def solve_optimization_task(args):
    """
    Worker function for parallel execution. Solves the MIP model for a specific p, k, q.
    """
    pass


# ==============================================================================
# 1. SUPPORT FUNCTIONS (Independent Modules)
# ==============================================================================
def load_data_files(files_cfg: dict) -> tuple:
    """Loads project CSV files into Pandas DataFrames."""
    wind_df = pd.read_csv(files_cfg["wind_file"], index_col="Time")
    sources_df = pd.read_csv(files_cfg["sources_file"])
    sensors_df = pd.read_csv(files_cfg["sensors_file"])
    scenarios_df = pd.read_csv(files_cfg["scenarios_file"])

    return wind_df, sources_df, sensors_df, scenarios_df


def simulate_plumes(
    sources_df: pd.DataFrame, wind_df: pd.DataFrame, grid_cfg: dict
) -> pd.DataFrame:
    """Builds the spatial grid and runs the Gaussian Plume model to extract signal data."""
    xar = np.arange(0, grid_cfg["x_size"], grid_cfg["dx"])
    yar = np.arange(0, grid_cfg["y_size"], grid_cfg["dy"])
    zar = np.arange(0, grid_cfg["z_size"], grid_cfg["dz"])

    grid = chama.simulation.Grid(xar, yar, zar)
    signal = pd.DataFrame()

    for _, row in sources_df.iterrows():
        scenario_name = row["Scenario"]
        source = chama.simulation.Source(row["X"], row["Y"], row["Z"], row["Leakrate"])
        gauss_plume = chama.simulation.GaussianPlume(grid, source, wind_df)
        gauss_plume.run()
        conc = gauss_plume.conc.rename(columns={"S": scenario_name})

        if signal.empty:
            signal = conc
        else:
            signal[scenario_name] = conc[scenario_name]

    # Save raw plume data for future analysis or plotting
    plume_output_file = Path("plumes_signal_output.csv")
    signal.to_csv(plume_output_file, index=False, sep=";", decimal=",")

    return signal


def calculate_impacts(
    signal: pd.DataFrame, sensors_df: pd.DataFrame, grid_cfg: dict
) -> tuple:
    """Cross-references plumes with sensor positions to determine detection times."""
    pass


def dispatch_optimizations(
    min_det_time: pd.DataFrame,
    sensor_chars: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    opt_cfg: dict,
) -> list:
    """Creates the task queue (p, k, q) and distributes it across CPU cores."""
    pass


def export_optimization_results(results_list: list):
    """Sorts the results list and saves the final CSV file."""
    pass


# ==============================================================================
# 2. MAIN ORCHESTRATOR
# ==============================================================================
def run_optimization_pipeline(files_cfg: dict, opt_cfg: dict, grid_cfg: dict):
    """
    Main function that orchestrates data flow between engine submodules.
    """
    # 1. Data Loading
    wind_df, sources_df, sensors_df, scenarios_df = load_data_files(files_cfg)

    # 2. Physical Simulation
    signal_df = simulate_plumes(sources_df, wind_df, grid_cfg)
