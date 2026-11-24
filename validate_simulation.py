from pathlib import Path

import numpy as np
import pandas as pd

# Find the dispatch results CSV beside this Python file
project_folder = Path(__file__).resolve().parent
results_file = project_folder / "microgrid_dispatch_results.csv"

if not results_file.exists():
    raise FileNotFoundError(
        "microgrid_dispatch_results.csv was not found. "
        "Run microgrid_dispatch.py first."
    )

# Read the battery and grid simulation results
results = pd.read_csv(results_file)

# Check that the results file has all required columns
required_columns = {
    "hour",
    "solar_generation_kw",
    "demand_kw",
    "battery_charge_power_kw",
    "battery_discharge_power_kw",
    "grid_import_power_kw",
    "grid_export_power_kw",
    "battery_energy_kwh",
    "battery_soc_percent",
}

missing_columns = required_columns - set(results.columns)

if missing_columns:
    raise ValueError(
        f"The results CSV is missing these columns: "
        f"{sorted(missing_columns)}"
    )

# These settings must match the settings in microgrid_dispatch.py
battery_capacity_kwh = 20.0
initial_soc_percent = 20.0
minimum_soc_percent = 20.0
maximum_soc_percent = 100.0

maximum_charge_power_kw = 5.0
maximum_discharge_power_kw = 5.0

charging_efficiency = 0.95
discharging_efficiency = 0.95
time_step_hours = 1.0

tolerance = 1e-6

# Convert the results columns into NumPy arrays for checking
hours = results["hour"].to_numpy(dtype=float)
solar_power = results["solar_generation_kw"].to_numpy(dtype=float)
demand_power = results["demand_kw"].to_numpy(dtype=float)
charge_power = results["battery_charge_power_kw"].to_numpy(dtype=float)
discharge_power = results["battery_discharge_power_kw"].to_numpy(dtype=float)
grid_import = results["grid_import_power_kw"].to_numpy(dtype=float)
grid_export = results["grid_export_power_kw"].to_numpy(dtype=float)
battery_energy = results["battery_energy_kwh"].to_numpy(dtype=float)
battery_soc = results["battery_soc_percent"].to_numpy(dtype=float)

checks = []

def add_check(name, passed, detail):
    """Save the result of one validation check."""
    checks.append((name, bool(passed), detail))


# Check that this is a 24-hour simulation with hours 0 to 23
correct_row_count = len(results) == 24
add_check(
    "24 hourly records",
    correct_row_count,
    f"Found {len(results)} rows; expected 24."
)

correct_hours = (
    correct_row_count
    and np.array_equal(hours, np.arange(24, dtype=float))
)
add_check(
    "Hours run from 0 to 23",
    correct_hours,
    "The hour values should appear once each, in order."
)

# Check that the battery stays within its SOC limits
soc_is_valid = np.all(
    (battery_soc >= minimum_soc_percent - tolerance)
    & (battery_soc <= maximum_soc_percent + tolerance)
)
add_check(
    "Battery SOC limits",
    soc_is_valid,
    f"Observed range: {battery_soc.min():.2f}% to "
    f"{battery_soc.max():.2f}%."
)

# Check that stored energy stays within the battery's capacity
minimum_energy_kwh = (
    battery_capacity_kwh * minimum_soc_percent / 100
)
maximum_energy_kwh = (
    battery_capacity_kwh * maximum_soc_percent / 100
)

energy_is_valid = np.all(
    (battery_energy >= minimum_energy_kwh - tolerance)
    & (battery_energy <= maximum_energy_kwh + tolerance)
)
add_check(
    "Battery energy limits",
    energy_is_valid,
    f"Allowed range: {minimum_energy_kwh:.2f} to "
    f"{maximum_energy_kwh:.2f} kWh."
)

# Check charge and discharge power ratings
charge_power_is_valid = np.all(
    (charge_power >= -tolerance)
    & (charge_power <= maximum_charge_power_kw + tolerance)
)
add_check(
    "Maximum charging power",
    charge_power_is_valid,
    f"Maximum allowed: {maximum_charge_power_kw:.2f} kW; "
    f"observed: {charge_power.max():.2f} kW."
)

discharge_power_is_valid = np.all(
    (discharge_power >= -tolerance)
    & (discharge_power <= maximum_discharge_power_kw + tolerance)
)
add_check(
    "Maximum discharging power",
    discharge_power_is_valid,
    f"Maximum allowed: {maximum_discharge_power_kw:.2f} kW; "
    f"observed: {discharge_power.max():.2f} kW."
)

# Check that grid imports and exports are never negative
grid_power_is_valid = np.all(
    (grid_import >= -tolerance)
    & (grid_export >= -tolerance)
)
add_check(
    "Grid import/export values are non-negative",
    grid_power_is_valid,
    "Grid import and export should each be zero or positive."
)

# Check electrical power balance at the AC bus for every hour:
# solar + battery discharge + grid import
#     = demand + battery charge + grid export
power_balance_residual = (
    solar_power
    + discharge_power
    + grid_import
    - demand_power
    - charge_power
    - grid_export
)

maximum_power_error_kw = float(
    np.max(np.abs(power_balance_residual))
)

power_balance_is_valid = maximum_power_error_kw <= tolerance
add_check(
    "Hourly electrical power balance",
    power_balance_is_valid,
    f"Maximum balance error: {maximum_power_error_kw:.8f} kW."
)

# Check battery energy changes from one hour to the next.
# Charging adds energy after charging losses.
# Discharging removes extra energy to account for discharge losses.
initial_battery_energy_kwh = (
    battery_capacity_kwh * initial_soc_percent / 100
)

previous_energy = np.concatenate(
    ([initial_battery_energy_kwh], battery_energy[:-1])
)

expected_battery_energy = (
    previous_energy
    + charge_power * time_step_hours * charging_efficiency
    - discharge_power * time_step_hours / discharging_efficiency
)

battery_energy_error = np.abs(
    battery_energy - expected_battery_energy
)

maximum_energy_error_kwh = float(np.max(battery_energy_error))

battery_energy_is_valid = maximum_energy_error_kwh <= tolerance
add_check(
    "Battery energy changes include efficiency losses",
    battery_energy_is_valid,
    f"Maximum energy calculation error: "
    f"{maximum_energy_error_kwh:.8f} kWh."
)

# Check that the SOC percentages match the recorded battery energy
expected_soc = battery_energy / battery_capacity_kwh * 100
maximum_soc_error_percent = float(
    np.max(np.abs(battery_soc - expected_soc))
)

soc_matches_energy = maximum_soc_error_percent <= tolerance
add_check(
    "SOC matches stored battery energy",
    soc_matches_energy,
    f"Maximum SOC calculation error: "
    f"{maximum_soc_error_percent:.8f} percentage points."
)

# Print every check
print("\nMICROGRID SIMULATION VALIDATION")
print("--------------------------------")

for name, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")

failed_checks = sum(not passed for _, passed, _ in checks)

if failed_checks == 0:
    print("\nResult: All validation checks passed.")
else:
    print(f"\nResult: {failed_checks} validation check(s) failed.")