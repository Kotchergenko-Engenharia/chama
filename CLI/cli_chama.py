import typer
from rich.console import Console

app = typer.Typer(
    help="CLI for Sensor Placement Optimization",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# TODO
@app.command()
def main():
    pass


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
