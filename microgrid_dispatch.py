from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# Find the CSV beside this Python file
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "microgrid_hourly_data.csv"
output_file = project_folder / "microgrid_dispatch_results.csv"

# Check that the Lesson 10 CSV exists
if not input_file.exists():
    raise FileNotFoundError(
        "microgrid_hourly_data.csv was not found. "
        "Run data_analysis.py first."
    )

# Read the solar and demand data
hourly_data = pd.read_csv(input_file)

# Check that the CSV has the columns this simulation needs
required_columns = {
    "hour",
    "solar_generation_kw",
    "demand_kw",
}

missing_columns = required_columns - set(hourly_data.columns)

if missing_columns:
    raise ValueError(
        f"The CSV is missing these columns: {sorted(missing_columns)}"
    )

# Battery and simulation settings
battery_capacity_kwh = 20.0
initial_soc_percent = 20.0
minimum_soc_percent = 20.0
maximum_soc_percent = 100.0

maximum_charge_power_kw = 5.0
maximum_discharge_power_kw = 5.0

charging_efficiency = 0.95
discharging_efficiency = 0.95
time_step_hours = 1.0

# Convert battery SOC limits to energy in kWh
minimum_battery_energy_kwh = (
    battery_capacity_kwh * minimum_soc_percent / 100
)
maximum_battery_energy_kwh = (
    battery_capacity_kwh * maximum_soc_percent / 100
)
battery_energy_kwh = (
    battery_capacity_kwh * initial_soc_percent / 100
)

# Store one result for each hour
dispatch_records = []

# Simulate the system chronologically, one hour at a time
for row in hourly_data.itertuples(index=False):
    hour = int(row.hour)
    solar_power_kw = float(row.solar_generation_kw)
    demand_power_kw = float(row.demand_kw)

    # Positive balance means surplus; negative means a deficit
    power_balance_kw = solar_power_kw - demand_power_kw

    battery_charge_power_kw = 0.0
    battery_discharge_power_kw = 0.0
    grid_import_power_kw = 0.0
    grid_export_power_kw = 0.0

    if power_balance_kw > 0:
        # Surplus solar is available to charge the battery
        surplus_power_kw = power_balance_kw

        # Limit charging by available solar, battery power rating,
        # and the battery's remaining storage capacity
        remaining_capacity_kwh = (
            maximum_battery_energy_kwh - battery_energy_kwh
        )

        maximum_charge_allowed_by_capacity_kw = max(
            0.0,
            remaining_capacity_kwh
            / (charging_efficiency * time_step_hours),
        )

        battery_charge_power_kw = min(
            surplus_power_kw,
            maximum_charge_power_kw,
            maximum_charge_allowed_by_capacity_kw,
        )

        # Charging losses mean stored energy increases by less
        battery_energy_kwh += (
            battery_charge_power_kw
            * time_step_hours
            * charging_efficiency
        )

        # Keep the battery within its maximum energy limit
        battery_energy_kwh = min(
            battery_energy_kwh,
            maximum_battery_energy_kwh,
        )

        # Export any surplus the battery could not store
        grid_export_power_kw = (
            surplus_power_kw - battery_charge_power_kw
        )

    elif power_balance_kw < 0:
        # There is a deficit after solar has served the load
        deficit_power_kw = -power_balance_kw

        # Limit discharge by available battery energy and power rating
        available_battery_energy_kwh = max(
            0.0,
            battery_energy_kwh - minimum_battery_energy_kwh,
        )

        maximum_discharge_allowed_by_energy_kw = (
            available_battery_energy_kwh
            * discharging_efficiency
            / time_step_hours
        )

        battery_discharge_power_kw = min(
            deficit_power_kw,
            maximum_discharge_power_kw,
            maximum_discharge_allowed_by_energy_kw,
        )

        # More energy leaves the battery than reaches the load,
        # because discharge efficiency is less than 100%
        battery_energy_kwh -= (
            battery_discharge_power_kw
            * time_step_hours
            / discharging_efficiency
        )

        # Keep the battery within its minimum energy limit
        battery_energy_kwh = max(
            battery_energy_kwh,
            minimum_battery_energy_kwh,
        )

        # Import any deficit the battery could not supply
        grid_import_power_kw = (
            deficit_power_kw - battery_discharge_power_kw
        )

    # Calculate the battery state of charge after this hour
    battery_soc_percent = (
        battery_energy_kwh / battery_capacity_kwh * 100
    )

    dispatch_records.append({
        "hour": hour,
        "solar_generation_kw": solar_power_kw,
        "demand_kw": demand_power_kw,
        "power_balance_kw": power_balance_kw,
        "battery_charge_power_kw": battery_charge_power_kw,
        "battery_discharge_power_kw": battery_discharge_power_kw,
        "grid_import_power_kw": grid_import_power_kw,
        "grid_export_power_kw": grid_export_power_kw,
        "battery_energy_kwh": battery_energy_kwh,
        "battery_soc_percent": battery_soc_percent,
    })

# Turn the hourly results into a table
dispatch_results = pd.DataFrame(dispatch_records)

# Save the table beside this script
dispatch_results.to_csv(output_file, index=False)

# Calculate daily grid energy
total_grid_import_kwh = (
    dispatch_results["grid_import_power_kw"].sum()
    * time_step_hours
)
total_grid_export_kwh = (
    dispatch_results["grid_export_power_kw"].sum()
    * time_step_hours
)

final_soc_percent = (
    battery_energy_kwh / battery_capacity_kwh * 100
)

# Display the daily summary
print("\nRULE-BASED MICROGRID DISPATCH SUMMARY")
print("--------------------------------------")
print(f"Initial battery SOC: {initial_soc_percent:.1f}%")
print(f"Final battery SOC:   {final_soc_percent:.1f}%")
print(f"Grid energy imported: {total_grid_import_kwh:.2f} kWh")
print(f"Grid energy exported: {total_grid_export_kwh:.2f} kWh")
print(
    "Peak grid import: "
    f"{dispatch_results['grid_import_power_kw'].max():.2f} kW"
)

print("\nFIRST 10 HOURLY RESULTS")
print(
    dispatch_results.head(10).to_string(index=False)
)

print(f"\nSaved hourly results to: {output_file.name}")
# Plot battery SOC, including the starting SOC at time 0
soc_times = [0] + (
    dispatch_results["hour"] + time_step_hours
).tolist()

soc_values = (
    [initial_soc_percent]
    + dispatch_results["battery_soc_percent"].tolist()
)

plt.figure(figsize=(10, 5))
plt.plot(
    soc_times,
    soc_values,
    marker="o",
    label="Battery SOC"
)
plt.axhline(
    minimum_soc_percent,
    color="red",
    linestyle="--",
    label="Minimum SOC"
)
plt.axhline(
    maximum_soc_percent,
    color="green",
    linestyle="--",
    label="Maximum SOC"
)
plt.title("Battery State of Charge During the Day")
plt.xlabel("Hour of day")
plt.ylabel("Battery SOC (%)")
plt.xticks(range(0, 25, 2))
plt.xlim(0, 24)
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(project_folder / "battery_soc.png", dpi=150)

# Plot grid imports above zero and exports below zero
plot_hours = dispatch_results["hour"]

plt.figure(figsize=(10, 5))
plt.bar(
    plot_hours,
    dispatch_results["grid_import_power_kw"],
    label="Grid import",
    color="tab:blue"
)
plt.bar(
    plot_hours,
    -dispatch_results["grid_export_power_kw"],
    label="Grid export (shown below zero)",
    color="tab:orange"
)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Hourly Grid Imports and Exports")
plt.xlabel("Hour of day")
plt.ylabel("Grid power (kW)")
plt.xticks(range(0, 24, 2))
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(project_folder / "grid_power.png", dpi=150)

plt.show()