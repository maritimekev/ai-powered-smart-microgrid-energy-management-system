from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Find the input files beside this Python script
project_folder = Path(__file__).resolve().parent
baseline_file = project_folder / "microgrid_hourly_data.csv"
dispatch_file = project_folder / "microgrid_dispatch_results.csv"

# Check that the required input files exist
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

# Read the solar/demand profile and battery dispatch results
baseline = pd.read_csv(baseline_file)
dispatch = pd.read_csv(dispatch_file)

# Check that the CSV files contain the required columns
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
        f"The hourly data CSV is missing: {sorted(missing_baseline)}"
    )

if missing_dispatch:
    raise ValueError(
        f"The dispatch results CSV is missing: {sorted(missing_dispatch)}"
    )

# Check that both files describe the same 24 hours
hours = baseline["hour"].to_numpy(dtype=int)
dispatch_hours = dispatch["hour"].to_numpy(dtype=int)

if len(hours) != 24 or not np.array_equal(hours, np.arange(24)):
    raise ValueError(
        "Expected 24 hourly records, with hours numbered from 0 to 23."
    )

if not np.array_equal(hours, dispatch_hours):
    raise ValueError(
        "The hours in the two CSV files do not match."
    )

# Confirm that the battery results use the same solar and demand data
for column in ["solar_generation_kw", "demand_kw"]:
    if not np.allclose(
        baseline[column].to_numpy(dtype=float),
        dispatch[column].to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(
            f"The {column} data differs between the two CSV files."
        )

# Example import tariff in generic currency units (CU) per kWh.
# It is lower overnight and higher during the evening peak.
import_price_cu_per_kwh = np.array([
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.16, 0.16, 0.16, 0.16,
    0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,
    0.30, 0.30, 0.30, 0.30, 0.30,
    0.12, 0.12,
])

# Example export credit in CU per kWh
export_price_cu_per_kwh = np.full(24, 0.05)

# Each row is the average power for one hour
time_step_hours = 1.0

if len(import_price_cu_per_kwh) != 24:
    raise ValueError("The import-price profile must have 24 values.")

if len(export_price_cu_per_kwh) != 24:
    raise ValueError("The export-price profile must have 24 values.")

# Read solar generation and demand
solar_power_kw = baseline["solar_generation_kw"].to_numpy(dtype=float)
demand_power_kw = baseline["demand_kw"].to_numpy(dtype=float)

# Calculate grid exchange without a battery
no_battery_import_kw = np.maximum(
    demand_power_kw - solar_power_kw,
    0.0,
)

no_battery_export_kw = np.maximum(
    solar_power_kw - demand_power_kw,
    0.0,
)

# Read grid exchange from the battery simulation
with_battery_import_kw = dispatch[
    "grid_import_power_kw"
].to_numpy(dtype=float)

with_battery_export_kw = dispatch[
    "grid_export_power_kw"
].to_numpy(dtype=float)

# Convert power to hourly energy
no_battery_import_kwh = no_battery_import_kw * time_step_hours
no_battery_export_kwh = no_battery_export_kw * time_step_hours

with_battery_import_kwh = with_battery_import_kw * time_step_hours
with_battery_export_kwh = with_battery_export_kw * time_step_hours

# Calculate import charges and export credits for each hour
no_battery_import_cost_cu = (
    no_battery_import_kwh * import_price_cu_per_kwh
)
no_battery_export_credit_cu = (
    no_battery_export_kwh * export_price_cu_per_kwh
)

with_battery_import_cost_cu = (
    with_battery_import_kwh * import_price_cu_per_kwh
)
with_battery_export_credit_cu = (
    with_battery_export_kwh * export_price_cu_per_kwh
)

# Net grid cost = import charges minus export credits
no_battery_net_cost_cu = (
    no_battery_import_cost_cu - no_battery_export_credit_cu
)
with_battery_net_cost_cu = (
    with_battery_import_cost_cu - with_battery_export_credit_cu
)

# Sum the hourly values to get daily totals
no_battery_import_cost_total_cu = no_battery_import_cost_cu.sum()
no_battery_export_credit_total_cu = no_battery_export_credit_cu.sum()
no_battery_net_cost_total_cu = no_battery_net_cost_cu.sum()

with_battery_import_cost_total_cu = with_battery_import_cost_cu.sum()
with_battery_export_credit_total_cu = with_battery_export_credit_cu.sum()
with_battery_net_cost_total_cu = with_battery_net_cost_cu.sum()

# Positive savings means the battery case has a lower net grid cost
cost_reduction_cu = (
    no_battery_net_cost_total_cu
    - with_battery_net_cost_total_cu
)

if no_battery_net_cost_total_cu > 0:
    cost_reduction_percent = (
        cost_reduction_cu / no_battery_net_cost_total_cu * 100
    )
else:
    cost_reduction_percent = None

# Display the daily comparison
print("\nILLUSTRATIVE DAILY GRID-COST COMPARISON")
print("---------------------------------------")
print("Prices are examples in generic currency units (CU).")
print()
print(
    "No-battery import charge: "
    f"{no_battery_import_cost_total_cu:.2f} CU"
)
print(
    "No-battery export credit: "
    f"{no_battery_export_credit_total_cu:.2f} CU"
)
print(
    "No-battery net grid cost: "
    f"{no_battery_net_cost_total_cu:.2f} CU"
)
print()
print(
    "With-battery import charge: "
    f"{with_battery_import_cost_total_cu:.2f} CU"
)
print(
    "With-battery export credit: "
    f"{with_battery_export_credit_total_cu:.2f} CU"
)
print(
    "With-battery net grid cost: "
    f"{with_battery_net_cost_total_cu:.2f} CU"
)
print()
print(
    "Cost reduction (no-battery cost minus with-battery cost): "
    f"{cost_reduction_cu:.2f} CU"
)

if cost_reduction_percent is not None:
    print(f"Cost reduction percentage: {cost_reduction_percent:.1f}%")

# Save hourly prices, energy exchange, and costs to a CSV file
cost_results = pd.DataFrame({
    "hour": hours,
    "import_price_cu_per_kwh": import_price_cu_per_kwh,
    "export_price_cu_per_kwh": export_price_cu_per_kwh,
    "no_battery_import_kwh": no_battery_import_kwh,
    "no_battery_export_kwh": no_battery_export_kwh,
    "no_battery_import_cost_cu": no_battery_import_cost_cu,
    "no_battery_export_credit_cu": no_battery_export_credit_cu,
    "no_battery_net_cost_cu": no_battery_net_cost_cu,
    "with_battery_import_kwh": with_battery_import_kwh,
    "with_battery_export_kwh": with_battery_export_kwh,
    "with_battery_import_cost_cu": with_battery_import_cost_cu,
    "with_battery_export_credit_cu": with_battery_export_credit_cu,
    "with_battery_net_cost_cu": with_battery_net_cost_cu,
})

output_file = project_folder / "microgrid_cost_comparison.csv"
cost_results.to_csv(output_file, index=False)

# Draw a chart comparing net daily grid cost
labels = ["No battery", "With battery"]
net_costs = [
    no_battery_net_cost_total_cu,
    with_battery_net_cost_total_cu,
]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, net_costs, color=["gray", "seagreen"])
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Illustrative Daily Net Grid Cost")
plt.ylabel("Net grid cost (CU)")
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f} CU",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

chart_file = project_folder / "microgrid_cost_comparison.png"
plt.savefig(chart_file, dpi=150)

print(f"\nSaved hourly cost data to: {output_file.name}")
print(f"Saved cost chart to: {chart_file.name}")
