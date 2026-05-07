import warnings
import concurrent.futures
from pathlib import Path

import pandas as pd
import numpy as np

import chama
from chama.optimize import ImpactFormulation

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 0. PARALLEL WORKER
# ==============================================================================
def solve_optimization_task(args):
    """
    Worker function for parallel execution. Solves the MIP model for a specific p, k, q.
    """
    p, k, q, solver_name, df_impact, df_sensor, df_scenario = args

    model = ImpactFormulation(k=k, q=q)

    # Uses CHAMA's high-level wrapper
    res = model.solve(
        impact=df_impact,
        sensor=df_sensor,
        scenario=df_scenario,
        sensor_budget=p,
        mip_solver_name=solver_name,
    )

    time_hr = res["Objective"] / 3600
    print(f" [OK] p={p:02d}, k={k}, q={q:.2f} solved in {time_hr:.2f} expected hours")

    return (
        p,
        k,
        q,
        time_hr,
        res["Solved"],
        res["FractionDetected"],
        res["TotalSensorCost"],
        res["Sensors"],
    )


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
    tar = np.arange(0, grid_cfg["tsize"], grid_cfg["dt"])
    sensors_dict = {}

    for _, row in sensors_df.iterrows():
        s_name = row["Sensor"]
        loc = chama.sensors.Stationary(location=(row["X"], row["Y"], row["Z"]))
        pt = chama.sensors.Point(threshold=row["Threshold"], sample_times=tar)
        sensors_dict[s_name] = chama.sensors.Sensor(position=loc, detector=pt)

    det_times = chama.impact.extract_detection_times(signal, sensors_dict)
    det_time_stats = chama.impact.detection_time_stats(det_times)

    # Prepare exact format expected by Pyomo
    min_det_time = det_time_stats[["Scenario", "Sensor", "Min"]].copy()
    min_det_time.rename(columns={"Min": "Impact"}, inplace=True)
    sensor_chars = sensors_df[["Sensor", "Cost", "X", "Y", "Z"]]

    return min_det_time, sensor_chars


def dispatch_optimizations(
    min_det_time: pd.DataFrame,
    sensor_chars: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    opt_cfg: dict,
) -> list:
    """Creates the task queue (p, k, q) and distributes it across CPU cores."""
    tasks = []
    for p in opt_cfg["budget_list"]:
        for k_val in opt_cfg["k_list"]:
            for q_val in opt_cfg["q_list"]:
                tasks.append(
                    (
                        p,
                        k_val,
                        q_val,
                        opt_cfg["solver"],
                        min_det_time,
                        sensor_chars,
                        scenarios_df,
                    )
                )

    results_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for p, k_val, q_val, time_hr, selected_sensors in executor.map(
            solve_optimization_task, tasks
        ):
            sensors_str = " | ".join(selected_sensors) if selected_sensors else "None"
            results_list.append(
                {
                    "Budget (p)": p,
                    "Votes (k)": k_val,
                    "Failure Prob (q)": q_val,
                    "Expected Time (hr)": round(time_hr, 2),
                    "Selected Sensors": sensors_str,
                }
            )

    return results_list


def export_optimization_results(results_list: list):
    """Sorts the results list and saves the final CSV file."""
    output_df = pd.DataFrame(results_list)
    output_df.sort_values(
        by=["Votes (k)", "Failure Prob (q)", "Budget (p)"], inplace=True
    )

    results_output_file = Path("optimization_results.csv")
    output_df.to_csv(results_output_file, index=False, sep=";", decimal=",")


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

    # 3. Impact Modeling
    min_det_time, sensor_chars = calculate_impacts(signal_df, sensors_df, grid_cfg)

    # 4. Mathematical Resolution (MIP)
    results_list = dispatch_optimizations(
        min_det_time, sensor_chars, scenarios_df, opt_cfg
    )

    # 5. Export
    export_optimization_results(results_list)
