from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Locate both CSV files in the same folder as this Python file
project_folder = Path(__file__).resolve().parent
baseline_file = project_folder / "microgrid_hourly_data.csv"
dispatch_file = project_folder / "microgrid_dispatch_results.csv"

# Stop with a clear message if either input file is missing
if not baseline_file.exists():
    raise FileNotFoundError(
        "microgrid_hourly_data.csv was not found. "
        "Run data_analysis.py first."
    )

if not dispatch_file.exists():
    raise FileNotFoundError(
        "microgrid_dispatch_results.csv was not found. "
        "Run microgrid_dispatch.py first."
    )

# Load the two sets of results
baseline = pd.read_csv(baseline_file)
dispatch = pd.read_csv(dispatch_file)

# Check that both files contain the columns we need
required_baseline_columns = {
    "hour",
    "solar_generation_kw",
    "demand_kw",
}

required_dispatch_columns = {
    "hour",
    "solar_generation_kw",
    "demand_kw",
    "grid_import_power_kw",
    "grid_export_power_kw",
}

missing_baseline = required_baseline_columns - set(baseline.columns)
missing_dispatch = required_dispatch_columns - set(dispatch.columns)

if missing_baseline:
    raise ValueError(
        f"The baseline CSV is missing: {sorted(missing_baseline)}"
    )

if missing_dispatch:
    raise ValueError(
        f"The dispatch CSV is missing: {sorted(missing_dispatch)}"
    )

# Confirm that the two files contain matching hourly inputs
if not np.array_equal(
    baseline["hour"].to_numpy(),
    dispatch["hour"].to_numpy()
):
    raise ValueError(
        "The hour values do not match between the two CSV files."
    )

for column in ["solar_generation_kw", "demand_kw"]:
    if not np.allclose(
        baseline[column].to_numpy(),
        dispatch[column].to_numpy(),
        rtol=0,
        atol=1e-9
    ):
        raise ValueError(
            f"The {column} values differ between the CSV files."
        )

# Each row represents one hour in this simulation
time_step_hours = 1.0

solar_power_kw = baseline["solar_generation_kw"].to_numpy()
demand_power_kw = baseline["demand_kw"].to_numpy()

# Calculate the grid exchange if there were no battery
no_battery_import_kw = np.maximum(
    demand_power_kw - solar_power_kw,
    0.0
)

no_battery_export_kw = np.maximum(
    solar_power_kw - demand_power_kw,
    0.0
)

no_battery_import_kwh = (
    no_battery_import_kw.sum() * time_step_hours
)
no_battery_export_kwh = (
    no_battery_export_kw.sum() * time_step_hours
)

# Get grid exchange from the battery simulation
battery_import_kwh = (
    dispatch["grid_import_power_kw"].sum() * time_step_hours
)
battery_export_kwh = (
    dispatch["grid_export_power_kw"].sum() * time_step_hours
)

# Compare grid imports and net grid energy
gross_import_reduction_kwh = (
    no_battery_import_kwh - battery_import_kwh
)

if no_battery_import_kwh > 0:
    gross_import_reduction_percent = (
        gross_import_reduction_kwh
        / no_battery_import_kwh
        * 100
    )
else:
    gross_import_reduction_percent = 0.0

no_battery_net_grid_kwh = (
    no_battery_import_kwh - no_battery_export_kwh
)
battery_net_grid_kwh = (
    battery_import_kwh - battery_export_kwh
)
change_in_net_grid_kwh = (
    battery_net_grid_kwh - no_battery_net_grid_kwh
)

# Display the comparison
print("\nDAILY PERFORMANCE COMPARISON")
print("----------------------------")
print(
    f"No-battery grid import:       "
    f"{no_battery_import_kwh:.2f} kWh"
)
print(
    f"With-battery grid import:     "
    f"{battery_import_kwh:.2f} kWh"
)
print(
    f"Gross import reduction:       "
    f"{gross_import_reduction_kwh:.2f} kWh "
    f"({gross_import_reduction_percent:.1f}%)"
)
print()
print(
    f"No-battery grid export:       "
    f"{no_battery_export_kwh:.2f} kWh"
)
print(
    f"With-battery grid export:     "
    f"{battery_export_kwh:.2f} kWh"
)
print()
print(
    f"No-battery net grid import:   "
    f"{no_battery_net_grid_kwh:.2f} kWh"
)
print(
    f"With-battery net grid import: "
    f"{battery_net_grid_kwh:.2f} kWh"
)
print(
    f"Change in net grid import:    "
    f"{change_in_net_grid_kwh:.2f} kWh"
)
print(
    "\nNet grid import means grid import minus grid export. "
    "A positive change means the battery case uses more net "
    "grid energy over the day."
)

# Make a chart comparing daily imports and exports
categories = ["Grid import", "Grid export"]
positions = np.arange(len(categories))
bar_width = 0.35

no_battery_values = [
    no_battery_import_kwh,
    no_battery_export_kwh,
]
with_battery_values = [
    battery_import_kwh,
    battery_export_kwh,
]

plt.figure(figsize=(9, 5))
plt.bar(
    positions - bar_width / 2,
    no_battery_values,
    bar_width,
    label="No battery"
)
plt.bar(
    positions + bar_width / 2,
    with_battery_values,
    bar_width,
    label="With battery"
)

plt.title("Daily Grid Energy: No Battery vs With Battery")
plt.xlabel("Grid exchange")
plt.ylabel("Energy (kWh)")
plt.xticks(positions, categories)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the chart in the project folder
chart_file = project_folder / "performance_comparison.png"
plt.savefig(chart_file, dpi=150)

print(f"\nSaved comparison chart to: {chart_file.name}")

plt.show()