import numpy as np
import pandas as pd

# Solar PV system parameters
pv_rated_power = 10.0
standard_irradiance = 1000.0
noct = 45.0
temperature_coefficient = -0.004
inverter_efficiency = 0.96
time_step_hours = 1.0

# Hourly weather data
solar_irradiance = np.array([
    0, 0, 0, 0, 0, 50,
    150, 300, 500, 700, 850, 950,
    1000, 950, 850, 700, 500, 300,
    150, 50, 0, 0, 0, 0
], dtype=float)

ambient_temperature = np.array([
    22, 21, 21, 20, 20, 21,
    23, 25, 27, 29, 31, 33,
    35, 36, 35, 33, 31, 29,
    27, 25, 24, 23, 22, 22
], dtype=float)

# Hourly electricity demand (kW)
electricity_demand = np.array([
    2.0, 1.8, 1.7, 1.7,
    1.8, 2.2, 3.5, 5.0,
    4.5, 3.8, 3.5, 3.7,
    4.2, 4.0, 3.8, 4.0,
    4.5, 5.5, 7.0, 8.0,
    7.5, 6.0, 4.0, 2.8
], dtype=float)

# Check that every profile has 24 hourly values
if not (
    len(solar_irradiance)
    == len(ambient_temperature)
    == len(electricity_demand)
    == 24
):
    raise ValueError("Each hourly profile must contain 24 values.")

hours = np.arange(24)

# Estimate solar cell temperature for each hour
cell_temperature = ambient_temperature + (
    (noct - 20.0) / 800.0
) * solar_irradiance

# Calculate DC and AC solar power
dc_power = (
    pv_rated_power
    * (solar_irradiance / standard_irradiance)
    * (
        1 + temperature_coefficient
        * (cell_temperature - 25.0)
    )
)

dc_power = np.maximum(dc_power, 0.0)
solar_generation = dc_power * inverter_efficiency

# Positive balance is surplus; negative balance is a deficit
power_balance = solar_generation - electricity_demand

# Put the hourly data into a Pandas table
results = pd.DataFrame({
    "hour": hours,
    "irradiance_w_m2": solar_irradiance,
    "ambient_temperature_c": ambient_temperature,
    "cell_temperature_c": cell_temperature,
    "solar_generation_kw": solar_generation,
    "demand_kw": electricity_demand,
    "power_balance_kw": power_balance,
})

# Convert hourly average power to energy
results["solar_energy_kwh"] = (
    results["solar_generation_kw"] * time_step_hours
)
results["demand_energy_kwh"] = (
    results["demand_kw"] * time_step_hours
)

# Separate surplus and deficit
results["surplus_kw"] = np.maximum(
    results["power_balance_kw"], 0.0
)
results["deficit_kw"] = np.maximum(
    -results["power_balance_kw"], 0.0
)

# Calculate daily summary values
solar_energy = results["solar_energy_kwh"].sum()
demand_energy = results["demand_energy_kwh"].sum()

peak_solar_hour = int(np.argmax(solar_generation))
peak_demand_hour = int(np.argmax(electricity_demand))

print("\nMICROGRID DAILY SUMMARY")
print("-----------------------")
print(f"Solar energy: {solar_energy:.2f} kWh")
print(f"Demand energy: {demand_energy:.2f} kWh")
print(f"Net energy balance: {solar_energy - demand_energy:.2f} kWh")
print(
    f"Peak solar: {solar_generation[peak_solar_hour]:.2f} kW "
    f"at {peak_solar_hour:02d}:00"
)
print(
    f"Peak demand: {electricity_demand[peak_demand_hour]:.2f} kW "
    f"at {peak_demand_hour:02d}:00"
)

print("\nFIRST SIX HOURLY ROWS")
print(results.head(6).to_string(index=False))

# Save the hourly table as a CSV file
results.to_csv("microgrid_hourly_data.csv", index=False)
print("\nSaved results to microgrid_hourly_data.csv")