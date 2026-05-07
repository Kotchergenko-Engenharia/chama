import configparser
from pathlib import Path

import typer
from rich.console import Console

from typing import Annotated

app = typer.Typer(
    help="CLI for Sensor Placement Optimization",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Path to the configuration file",
        ),
    ],
):
    r"""
    Reads the provided text file, validates parameters, and triggers the CHAMA optimization engine.
    """
    # ---------------------------------------------------------
    # 1. PARSING AND VALIDATION
    # ---------------------------------------------------------
    config = configparser.ConfigParser()
    try:
        config.read(input_file, encoding="utf-8")
    except Exception as e:
        console.print(f"[bold red]Error reading file:[/bold red] {e}")
        raise typer.Exit(code=1)

    try:
        # [Files]
        files_cfg = {
            "wind_file": config.get("Files", "wind_file"),
            "sources_file": config.get("Files", "sources_file"),
            "sensors_file": config.get("Files", "sensors_file"),
            "scenarios_file": config.get("Files", "scenarios_file"),
        }

    except (configparser.NoOptionError, configparser.NoSectionError) as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    # File Existence Verification
    for file_label, file_path in files_cfg.items():
        if not Path(file_path).is_file():
            console.print(
                f"[bold red]File Not Found Error:[/bold red] The file '{file_path}' mapped to {file_label} does not exist."
            )
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

# ===================================================================
# INPUT FILES CONFIGURATION
# ===================================================================

# 1. wind.csv (Meteorological Data)
# Description: This file is injected into the Gaussian Plume model.
# Time -> Time in seconds since the start of the simulation.
# Wind Direction -> Wind direction in degrees.
# Wind Speed -> Wind speed in meters per second (m/s).
# Stability Class -> Pasquill-Gifford stability class (A - F), which defines atmospheric turbulence:
#   A - Extremely Unstable: Strong insolation, light winds (daytime).
#   B - Moderately Unstable: Moderate insolation.
#   C - Slightly Unstable: Slight insolation.
#   D - Neutral: Overcast sky or strong winds, regardless of the time of day.
#   E - Slightly Stable: Nighttime with few clouds.
#   F - Moderately Stable: Clear nights, light winds (common during thermal inversions).

# 2. source.csv (Leak Sources)
# Description: Defines where the leak(s) occur and their mass flow rate.
# Scenario -> UNIQUE scenario name. This links the source to the impact table.
# X, Y, Z -> Source coordinates (meters).
# Leakrate -> Continuous leak rate (e.g., g/s or kg/s - unit consistency must be maintained across the project).

# 3. sensors.csv (Candidate Sensors)
# Description: Defines the properties of the available detectors.
# Sensor -> Unique name of the detector.
# X, Y, Z -> Sensor coordinates (meters).
# Threshold -> Minimum concentration required to trigger the sensor's alarm.
# Cost -> Financial or logistical cost of the sensor.

# 4. scenarios.csv (Scenario Metadata)
# Description: Specifies the penalty applied if the plume is not detected.
# Scenario -> Must perfectly match the Scenario column in the source.csv file.
# Undetected Impact -> Maximum penalty (e.g., final simulation time or a financial fine multiplier).
# Probability -> (Optional) The likelihood of the scenario occurring.

# ===================================================================
# GENERAL SIMULATION & OPTIMIZATION PARAMETERS
# ===================================================================

# Grid -> Simulation bounding box and resolution. (Example: x_range=(0, 100), dx=10, y_range=(0, 100), dy=10, z_range=(0, 10), dz=1)
# Budget (p) -> List of available budgets defining how many sensors can be purchased/installed. (Example: [5, 10, 15, 20])
# Votes (k) -> List of numbers of sensors that must independently detect the plume before the final alarm triggers (Example: [1, 2]).
# Failure Probability (q) -> List of probabilitys of a sensor failing to operate when exposed to the gas (Example: [0.0, 0.1, 0.2]).
# Solver -> The Mixed Integer Programming (MIP) solver used to run the optimization problem (Example: glpk, appsi_highs).
