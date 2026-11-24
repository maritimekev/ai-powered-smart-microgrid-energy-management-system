from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Find the input files beside this Python file
project_folder = Path(__file__).resolve().parent
profile_file = project_folder / "microgrid_hourly_data.csv"
price_file = project_folder / "microgrid_cost_comparison.csv"
rule_based_file = project_folder / "microgrid_dispatch_results.csv"

# Check that the files from earlier steps exist
for file_path in [profile_file, price_file, rule_based_file]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path.name} was not found. "
            "Run the earlier simulation and cost-analysis scripts first."
        )

# Read hourly solar/demand data, example prices, and rule-based results
profile = pd.read_csv(profile_file)
prices = pd.read_csv(price_file)
rule_based = pd.read_csv(rule_based_file)

# Check that the required columns exist
required_profile_columns = {
    "hour",
    "solar_generation_kw",
    "demand_kw",
}
required_price_columns = {
    "hour",
    "import_price_cu_per_kwh",
    "export_price_cu_per_kwh",
}
required_rule_columns = {
    "hour",
    "solar_generation_kw",
    "demand_kw",
    "grid_import_power_kw",
    "grid_export_power_kw",
}

missing_profile = required_profile_columns - set(profile.columns)
missing_prices = required_price_columns - set(prices.columns)
missing_rule = required_rule_columns - set(rule_based.columns)

if missing_profile:
    raise ValueError(
        f"The hourly profile CSV is missing: {sorted(missing_profile)}"
    )
if missing_prices:
    raise ValueError(
        f"The price CSV is missing: {sorted(missing_prices)}"
    )
if missing_rule:
    raise ValueError(
        f"The rule-based results CSV is missing: {sorted(missing_rule)}"
    )

# This example models one day with 24 one-hour intervals
number_of_hours = len(profile)

if number_of_hours != 24:
    raise ValueError(
        f"Expected 24 hourly rows, but found {number_of_hours}."
    )

hours = profile["hour"].to_numpy(dtype=int)
price_hours = prices["hour"].to_numpy(dtype=int)
rule_hours = rule_based["hour"].to_numpy(dtype=int)

expected_hours = np.arange(24)

if not np.array_equal(hours, expected_hours):
    raise ValueError("Profile hours should run from 0 to 23.")
if not np.array_equal(price_hours, expected_hours):
    raise ValueError("Price hours should run from 0 to 23.")
if not np.array_equal(rule_hours, expected_hours):
    raise ValueError("Rule-based dispatch hours should run from 0 to 23.")

# Read solar generation, demand, and example tariffs
solar_power_kw = profile["solar_generation_kw"].to_numpy(dtype=float)
demand_power_kw = profile["demand_kw"].to_numpy(dtype=float)

import_price = prices[
    "import_price_cu_per_kwh"
].to_numpy(dtype=float)
export_price = prices[
    "export_price_cu_per_kwh"
].to_numpy(dtype=float)

# Confirm that the rule-based results use the same solar and demand profiles
if not np.allclose(
    solar_power_kw,
    rule_based["solar_generation_kw"].to_numpy(dtype=float),
    rtol=0,
    atol=1e-9,
):
    raise ValueError("Solar-generation profiles do not match.")

if not np.allclose(
    demand_power_kw,
    rule_based["demand_kw"].to_numpy(dtype=float),
    rtol=0,
    atol=1e-9,
):
    raise ValueError("Demand profiles do not match.")

# Simulation timestep
time_step_hours = 1.0

# Battery settings: keep these consistent with microgrid_dispatch.py
battery_capacity_kwh = 20.0
initial_soc_percent = 20.0
minimum_soc_percent = 20.0
maximum_soc_percent = 100.0

maximum_charge_power_kw = 5.0
maximum_discharge_power_kw = 5.0

charging_efficiency = 0.95
discharging_efficiency = 0.95

# Convert SOC limits into stored energy
initial_battery_energy_kwh = (
    battery_capacity_kwh * initial_soc_percent / 100
)
minimum_battery_energy_kwh = (
    battery_capacity_kwh * minimum_soc_percent / 100
)
maximum_battery_energy_kwh = (
    battery_capacity_kwh * maximum_soc_percent / 100
)

# The linear program has five decision variables for every hour:
# 1. Grid import power
# 2. Grid export power
# 3. Battery charging power
# 4. Battery discharging power
# 5. Battery energy at the end of the hour
#
# The index arrays identify where each group is stored in the solver vector.
grid_import_index = np.arange(0, number_of_hours)
grid_export_index = np.arange(
    number_of_hours, 2 * number_of_hours
)
charge_index = np.arange(
    2 * number_of_hours, 3 * number_of_hours
)
discharge_index = np.arange(
    3 * number_of_hours, 4 * number_of_hours
)
battery_energy_index = np.arange(
    4 * number_of_hours, 5 * number_of_hours
)

number_of_variables = 5 * number_of_hours

# The objective is total grid cost:
# import charges minus export credits.
objective = np.zeros(number_of_variables)

objective[grid_import_index] = (
    import_price * time_step_hours
)
objective[grid_export_index] = (
    -export_price * time_step_hours
)

# Build the hourly electrical power-balance equations and
# battery-energy equations.
number_of_equalities = 2 * number_of_hours
A_eq = np.zeros((number_of_equalities, number_of_variables))
b_eq = np.zeros(number_of_equalities)

for hour_index in range(number_of_hours):
    # Electrical power balance:
    # solar + battery discharge + grid import
    #     = demand + battery charge + grid export
    power_row = hour_index

    A_eq[power_row, grid_import_index[hour_index]] = 1.0
    A_eq[power_row, grid_export_index[hour_index]] = -1.0
    A_eq[power_row, charge_index[hour_index]] = -1.0
    A_eq[power_row, discharge_index[hour_index]] = 1.0

    b_eq[power_row] = (
        demand_power_kw[hour_index]
        - solar_power_kw[hour_index]
    )

    # Battery energy at the end of this hour:
    # current energy = previous energy
    #     + efficient charging
    #     - energy used for discharging
    energy_row = number_of_hours + hour_index

    A_eq[
        energy_row, battery_energy_index[hour_index]
    ] = 1.0
    A_eq[energy_row, charge_index[hour_index]] = (
        -charging_efficiency * time_step_hours
    )
    A_eq[energy_row, discharge_index[hour_index]] = (
        time_step_hours / discharging_efficiency
    )

    if hour_index == 0:
        # The first hour starts from the specified initial energy.
        b_eq[energy_row] = initial_battery_energy_kwh
    else:
        # Include the previous hour's ending battery energy.
        A_eq[
            energy_row, battery_energy_index[hour_index - 1]
        ] = -1.0

# Set limits for all variables:
# - Grid exchange cannot be negative.
# - Battery charge/discharge power has a maximum.
# - Battery energy remains within its SOC limits.
bounds = (
    [(0.0, None)] * number_of_hours
    + [(0.0, None)] * number_of_hours
    + [(0.0, maximum_charge_power_kw)] * number_of_hours
    + [(0.0, maximum_discharge_power_kw)] * number_of_hours
    + [
        (minimum_battery_energy_kwh, maximum_battery_energy_kwh)
    ] * number_of_hours
)

# Require the battery to finish at the same energy as it started.
# This prevents the optimizer from making the result look cheaper
# by simply emptying the battery at the end of the day.
bounds[battery_energy_index[-1]] = (
    initial_battery_energy_kwh,
    initial_battery_energy_kwh,
)

# Solve the linear optimization problem
optimization = linprog(
    c=objective,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=bounds,
    method="highs",
)

if not optimization.success:
    raise RuntimeError(
        "The optimizer could not find a solution: "
        f"{optimization.message}"
    )

# Extract the optimal hourly schedule
grid_import_kw = optimization.x[grid_import_index].copy()
grid_export_kw = optimization.x[grid_export_index].copy()
battery_charge_kw = optimization.x[charge_index].copy()
battery_discharge_kw = optimization.x[discharge_index].copy()
battery_energy_kwh = optimization.x[battery_energy_index].copy()

# Remove tiny numerical values such as 1e-12
for values in [
    grid_import_kw,
    grid_export_kw,
    battery_charge_kw,
    battery_discharge_kw,
]:
    values[np.abs(values) < 1e-9] = 0.0

battery_soc_percent = (
    battery_energy_kwh / battery_capacity_kwh * 100
)

# Check power balance for the optimized schedule
power_balance_error_kw = (
    solar_power_kw
    + battery_discharge_kw
    + grid_import_kw
    - demand_power_kw
    - battery_charge_kw
    - grid_export_kw
)
maximum_power_balance_error_kw = float(
    np.max(np.abs(power_balance_error_kw))
)

# Check battery energy transitions and efficiency calculations
previous_battery_energy_kwh = np.concatenate(
    (
        [initial_battery_energy_kwh],
        battery_energy_kwh[:-1],
    )
)

expected_battery_energy_kwh = (
    previous_battery_energy_kwh
    + battery_charge_kw
    * time_step_hours
    * charging_efficiency
    - battery_discharge_kw
    * time_step_hours
    / discharging_efficiency
)

maximum_battery_energy_error_kwh = float(
    np.max(
        np.abs(
            battery_energy_kwh
            - expected_battery_energy_kwh
        )
    )
)

if maximum_power_balance_error_kw > 1e-6:
    raise RuntimeError(
        "The optimized schedule failed the power-balance check."
    )

if maximum_battery_energy_error_kwh > 1e-6:
    raise RuntimeError(
        "The optimized schedule failed the battery-energy check."
    )

# Calculate the no-battery grid exchange
no_battery_import_kwh = (
    np.maximum(demand_power_kw - solar_power_kw, 0.0)
    * time_step_hours
)
no_battery_export_kwh = (
    np.maximum(solar_power_kw - demand_power_kw, 0.0)
    * time_step_hours
)

# Read the original rule-based grid exchange
rule_import_kwh = (
    rule_based["grid_import_power_kw"].to_numpy(dtype=float)
    * time_step_hours
)
rule_export_kwh = (
    rule_based["grid_export_power_kw"].to_numpy(dtype=float)
    * time_step_hours
)

# Calculate the daily grid costs for all three cases
no_battery_import_cost = np.sum(
    no_battery_import_kwh * import_price
)
no_battery_export_credit = np.sum(
    no_battery_export_kwh * export_price
)
no_battery_net_cost = (
    no_battery_import_cost - no_battery_export_credit
)

rule_import_cost = np.sum(rule_import_kwh * import_price)
rule_export_credit = np.sum(rule_export_kwh * export_price)
rule_net_cost = rule_import_cost - rule_export_credit

optimized_import_kwh = grid_import_kw * time_step_hours
optimized_export_kwh = grid_export_kw * time_step_hours

optimized_import_cost = np.sum(
    optimized_import_kwh * import_price
)
optimized_export_credit = np.sum(
    optimized_export_kwh * export_price
)
optimized_net_cost = (
    optimized_import_cost - optimized_export_credit
)

cost_saving_vs_rule = rule_net_cost - optimized_net_cost

if rule_net_cost > 0:
    cost_saving_percent_vs_rule = (
        cost_saving_vs_rule / rule_net_cost * 100
    )
else:
    cost_saving_percent_vs_rule = 0.0

# Check whether the solver scheduled charge and discharge in the same hour.
# A basic linear program does not strictly forbid this behavior.
simultaneous_charge_discharge = (
    (battery_charge_kw > 1e-7)
    & (battery_discharge_kw > 1e-7)
)

# Print the optimization summary
print("\nPRICE-AWARE OPTIMIZED DISPATCH")
print("--------------------------------")
print(f"Solver status: {optimization.message}")
print(
    f"Initial battery SOC: {initial_soc_percent:.1f}%"
)
print(
    f"Final battery SOC:   {battery_soc_percent[-1]:.1f}%"
)
print(
    f"Grid energy imported: "
    f"{optimized_import_kwh.sum():.2f} kWh"
)
print(
    f"Grid energy exported: "
    f"{optimized_export_kwh.sum():.2f} kWh"
)
print(
    f"Peak grid import: {grid_import_kw.max():.2f} kW"
)
print(
    f"Battery energy charged: "
    f"{(battery_charge_kw * time_step_hours).sum():.2f} kWh"
)
print(
    f"Battery energy discharged: "
    f"{(battery_discharge_kw * time_step_hours).sum():.2f} kWh"
)

print("\nNET GRID COST COMPARISON")
print("------------------------")
print(f"No battery:          {no_battery_net_cost:.2f} CU")
print(f"Rule-based battery:  {rule_net_cost:.2f} CU")
print(f"Optimized battery:   {optimized_net_cost:.2f} CU")
print(
    f"Saving vs rule-based: {cost_saving_vs_rule:.2f} CU "
    f"({cost_saving_percent_vs_rule:.1f}%)"
)

print("\nOPTIMIZATION CHECKS")
print("-------------------")
print(
    "Maximum power-balance error: "
    f"{maximum_power_balance_error_kw:.8f} kW"
)
print(
    "Maximum battery-energy error: "
    f"{maximum_battery_energy_error_kwh:.8f} kWh"
)

if np.any(simultaneous_charge_discharge):
    hours_with_simultaneous_operation = hours[
        simultaneous_charge_discharge
    ]
    print(
        "Warning: simultaneous charging and discharging was "
        "selected during hour(s): "
        f"{hours_with_simultaneous_operation.tolist()}"
    )
    print(
        "A mixed-integer model would be needed to prohibit this "
        "strictly."
    )
else:
    print("No simultaneous battery charging/discharging detected.")

# Save the optimized hourly schedule and costs
optimized_results = pd.DataFrame({
    "hour": hours,
    "solar_generation_kw": solar_power_kw,
    "demand_kw": demand_power_kw,
    "import_price_cu_per_kwh": import_price,
    "export_price_cu_per_kwh": export_price,
    "battery_charge_power_kw": battery_charge_kw,
    "battery_discharge_power_kw": battery_discharge_kw,
    "grid_import_power_kw": grid_import_kw,
    "grid_export_power_kw": grid_export_kw,
    "battery_energy_kwh": battery_energy_kwh,
    "battery_soc_percent": battery_soc_percent,
    "grid_import_energy_kwh": optimized_import_kwh,
    "grid_export_energy_kwh": optimized_export_kwh,
    "grid_import_cost_cu": (
        optimized_import_kwh * import_price
    ),
    "grid_export_credit_cu": (
        optimized_export_kwh * export_price
    ),
})

optimized_results["net_grid_cost_cu"] = (
    optimized_results["grid_import_cost_cu"]
    - optimized_results["grid_export_credit_cu"]
)

output_file = project_folder / "optimized_dispatch_results.csv"
optimized_results.to_csv(output_file, index=False)

# Plot battery SOC, grid exchange, and example prices
figure, axes = plt.subplots(3, 1, figsize=(11, 10))

# Add the initial SOC at time 0 so the graph shows the full day
soc_time = np.arange(number_of_hours + 1) * time_step_hours
soc_values = np.concatenate(
    ([initial_soc_percent], battery_soc_percent)
)

axes[0].plot(
    soc_time,
    soc_values,
    marker="o",
    label="Battery SOC",
)
axes[0].axhline(
    minimum_soc_percent,
    color="red",
    linestyle="--",
    label="Minimum SOC",
)
axes[0].set_title("Optimized Battery State of Charge")
axes[0].set_ylabel("SOC (%)")
axes[0].set_xticks(np.arange(0, 25, 2))
axes[0].set_xlim(0, 24)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Show exports below zero to distinguish them from imports
axes[1].bar(
    hours,
    grid_import_kw,
    label="Grid import",
    color="tab:blue",
)
axes[1].bar(
    hours,
    -grid_export_kw,
    label="Grid export (shown below zero)",
    color="tab:orange",
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Optimized Hourly Grid Exchange")
axes[1].set_ylabel("Power (kW)")
axes[1].set_xticks(np.arange(0, 24, 2))
axes[1].grid(True, axis="y", alpha=0.3)
axes[1].legend()

# Show the example import and export prices
axes[2].step(
    hours,
    import_price,
    where="mid",
    marker="o",
    label="Import price",
)
axes[2].step(
    hours,
    export_price,
    where="mid",
    marker="s",
    label="Export credit",
)
axes[2].set_title("Example Electricity Prices")
axes[2].set_xlabel("Hour of day")
axes[2].set_ylabel("Price (CU/kWh)")
axes[2].set_xticks(np.arange(0, 24, 2))
axes[2].grid(True, alpha=0.3)
axes[2].legend()

figure.tight_layout()

chart_file = project_folder / "optimized_dispatch.png"
figure.savefig(chart_file, dpi=150)

print(f"\nSaved optimized schedule to: {output_file.name}")
print(f"Saved optimization graph to: {chart_file.name}")

plt.show()