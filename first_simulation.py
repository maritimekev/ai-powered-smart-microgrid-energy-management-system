# ==========================================
# AI SMART MICROGRID
# LESSON 9: DATA VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt


# ==========================================
# 1. SOLAR PV SYSTEM PARAMETERS
# ==========================================

pv_rated_power = 10.0

standard_irradiance = 1000

noct = 45

temperature_coefficient = -0.004

inverter_efficiency = 0.96


# ==========================================
# 2. HOURLY WEATHER DATA
# ==========================================

solar_irradiance = [
    0, 0, 0, 0, 0, 50,
    150, 300, 500, 700, 850, 950,
    1000, 950, 850, 700, 500, 300,
    150, 50, 0, 0, 0, 0
]

ambient_temperature = [
    22, 21, 21, 20, 20, 21,
    23, 25, 27, 29, 31, 33,
    35, 36, 35, 33, 31, 29,
    27, 25, 24, 23, 22, 22
]


# ==========================================
# 3. ELECTRICITY DEMAND DATA
# ==========================================

electricity_demand = [
    2.0, 1.8, 1.7, 1.7,
    1.8, 2.2, 3.5, 5.0,
    4.5, 3.8, 3.5, 3.7,
    4.2, 4.0, 3.8, 4.0,
    4.5, 5.5, 7.0, 8.0,
    7.5, 6.0, 4.0, 2.8
]


# ==========================================
# 4. INITIALIZE SIMULATION DATA
# ==========================================

hours = list(range(24))

solar_generation = []

cell_temperatures = []

total_solar_energy = 0

total_demand_energy = 0


# ==========================================
# 5. CALCULATE SOLAR GENERATION
# ==========================================

for hour in range(24):

    irradiance = solar_irradiance[hour]

    ambient = ambient_temperature[hour]

    # Estimate solar cell temperature

    cell_temperature = ambient + (
        (noct - 20) / 800
    ) * irradiance

    cell_temperatures.append(cell_temperature)

    # Calculate DC solar power

    dc_power = pv_rated_power * (
        irradiance / standard_irradiance
    ) * (
        1 + temperature_coefficient * (
            cell_temperature - 25
        )
    )

    dc_power = max(0, dc_power)

    # Convert DC power into AC power

    ac_power = dc_power * inverter_efficiency

    solar_generation.append(ac_power)

    # Accumulate daily solar energy

    total_solar_energy += ac_power


# ==========================================
# 6. CALCULATE DAILY ELECTRICITY DEMAND
# ==========================================

for demand in electricity_demand:

    total_demand_energy += demand


# ==========================================
# 7. CALCULATE POWER BALANCE
# ==========================================

power_balance = []

for hour in range(24):

    balance = (
        solar_generation[hour]
        - electricity_demand[hour]
    )

    power_balance.append(balance)


# ==========================================
# 8. DAILY ENERGY SUMMARY
# ==========================================

print("\nSMART MICROGRID DAILY SUMMARY")

print("-----------------------------------")

print(f"Total Solar Energy: {total_solar_energy:.2f} kWh")

print(f"Total Demand Energy: {total_demand_energy:.2f} kWh")

print(f"Peak Demand: {max(electricity_demand):.2f} kW")

print(f"Peak Solar Generation: {max(solar_generation):.2f} kW")

print(f"Net Energy Balance: {total_solar_energy - total_demand_energy:.2f} kWh")

print("-----------------------------------")


# ==========================================
# 9. GRAPH 1: SOLAR GENERATION
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    hours,
    solar_generation,
    marker="o",
    label="Solar Generation"
)

plt.title("24-Hour Solar PV Generation Profile")

plt.xlabel("Time (Hours)")

plt.ylabel("Solar Power (kW)")

plt.xticks(range(0, 24, 2))

plt.grid(True)

plt.legend()

plt.tight_layout()


# ==========================================
# 10. GRAPH 2: ELECTRICITY DEMAND
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    hours,
    electricity_demand,
    marker="o",
    label="Electricity Demand"
)

plt.title("24-Hour Electricity Demand Profile")

plt.xlabel("Time (Hours)")

plt.ylabel("Demand (kW)")

plt.xticks(range(0, 24, 2))

plt.grid(True)

plt.legend()

plt.tight_layout()


# ==========================================
# 11. GRAPH 3: GENERATION VS DEMAND
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    hours,
    solar_generation,
    marker="o",
    label="Solar Generation"
)

plt.plot(
    hours,
    electricity_demand,
    marker="s",
    label="Electricity Demand"
)

plt.title("Solar Generation vs Electricity Demand")

plt.xlabel("Time (Hours)")

plt.ylabel("Power (kW)")

plt.xticks(range(0, 24, 2))

plt.grid(True)

plt.legend()

plt.tight_layout()


# ==========================================
# 12. DISPLAY ALL GRAPHS
# ==========================================

plt.show()