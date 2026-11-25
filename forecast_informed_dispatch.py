from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Find the forecast and weather files beside this script
project_folder = Path(__file__).resolve().parent
forecast_file = (
    project_folder / "nairobi_solar_forecast_2023_test_results.csv"
)
weather_file = (
    project_folder / "nairobi_nasa_power_2019_2023_hourly.csv"
)

for file_path in [forecast_file, weather_file]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path.name} was not found. "
            "Run the earlier data and forecasting scripts first."
        )

# ------------------------------------------------------------
# 1. LOAD AND CHECK THE ONE-HOUR-AHEAD FORECASTS
# ------------------------------------------------------------

forecast = pd.read_csv(forecast_file)

required_forecast_columns = {
    "forecast_timestamp_nairobi",
    "observed_solar_wh_m2",
    "persistence_prediction_wh_m2",
    "month_hour_average_prediction_wh_m2",
    "random_forest_prediction_wh_m2",
}

missing_forecast_columns = (
    required_forecast_columns - set(forecast.columns)
)

if missing_forecast_columns:
    raise ValueError(
        "The forecast results are missing columns: "
        f"{sorted(missing_forecast_columns)}"
    )

forecast["forecast_timestamp_nairobi"] = pd.to_datetime(
    forecast["forecast_timestamp_nairobi"],
    utc=True,
    errors="coerce",
).dt.tz_convert("Africa/Nairobi")

if forecast["forecast_timestamp_nairobi"].isna().any():
    raise ValueError("Some forecast timestamps could not be parsed.")

forecast = (
    forecast
    .sort_values("forecast_timestamp_nairobi")
    .reset_index(drop=True)
)

if len(forecast) != 8760:
    raise ValueError(
        f"Expected 8,760 hourly forecasts for 2023; "
        f"found {len(forecast)}."
    )

if forecast["forecast_timestamp_nairobi"].duplicated().any():
    raise ValueError("Duplicate forecast timestamps were found.")

forecast_times = forecast["forecast_timestamp_nairobi"]
expected_start = pd.Timestamp(
    "2023-01-01 00:00:00",
    tz="Africa/Nairobi",
)
expected_end = pd.Timestamp(
    "2023-12-31 23:00:00",
    tz="Africa/Nairobi",
)

if forecast_times.iloc[0] != expected_start:
    raise ValueError(
        "The forecast data does not start at Nairobi local "
        "midnight on January 1, 2023."
    )

if forecast_times.iloc[-1] != expected_end:
    raise ValueError(
        "The forecast data does not end at Nairobi local "
        "23:00 on December 31, 2023."
    )

time_gaps = forecast_times.diff().dropna()

if not (time_gaps == pd.Timedelta(hours=1)).all():
    raise ValueError(
        "The forecast data contains a missing or non-hourly timestamp."
    )

prediction_columns = {
    "persistence": "persistence_prediction_wh_m2",
    "month_hour_average": (
        "month_hour_average_prediction_wh_m2"
    ),
    "random_forest": "random_forest_prediction_wh_m2",
}

numeric_forecast_columns = [
    "observed_solar_wh_m2",
    *prediction_columns.values(),
]

for column in numeric_forecast_columns:
    forecast[column] = pd.to_numeric(
        forecast[column],
        errors="coerce",
    )

if forecast[numeric_forecast_columns].isna().any().any():
    raise ValueError(
        "The forecast file has missing or non-numeric solar values."
    )

# ------------------------------------------------------------
# 2. LOAD WEATHER TEMPERATURES
# ------------------------------------------------------------

weather = pd.read_csv(weather_file)

required_weather_columns = {
    "timestamp_nairobi",
    "allsky_sfc_sw_dwn",
    "t2m",
}

missing_weather_columns = (
    required_weather_columns - set(weather.columns)
)

if missing_weather_columns:
    raise ValueError(
        "The NASA weather file is missing columns: "
        f"{sorted(missing_weather_columns)}"
    )

weather["timestamp_nairobi"] = pd.to_datetime(
    weather["timestamp_nairobi"],
    utc=True,
    errors="coerce",
).dt.tz_convert("Africa/Nairobi")

weather["solar_resource_wh_m2"] = pd.to_numeric(
    weather["allsky_sfc_sw_dwn"],
    errors="coerce",
)

weather["ambient_temperature_c"] = pd.to_numeric(
    weather["t2m"],
    errors="coerce",
)

if weather[
    [
        "timestamp_nairobi",
        "solar_resource_wh_m2",
        "ambient_temperature_c",
    ]
].isna().any().any():
    raise ValueError(
        "The NASA weather file has missing or invalid values."
    )

if weather["timestamp_nairobi"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps were found in the NASA weather file."
    )

# Weather conditions for the forecasted target hour.
target_weather = weather[
    [
        "timestamp_nairobi",
        "solar_resource_wh_m2",
        "ambient_temperature_c",
    ]
].rename(columns={
    "timestamp_nairobi": "forecast_timestamp_nairobi",
    "solar_resource_wh_m2": "weather_solar_wh_m2",
    "ambient_temperature_c": "ambient_target_hour_c",
})

# The temperature known when the forecast is issued:
# the ambient temperature one hour before the target interval.
issue_weather = weather[
    ["timestamp_nairobi", "ambient_temperature_c"]
].copy()

issue_weather["forecast_timestamp_nairobi"] = (
    issue_weather["timestamp_nairobi"]
    + pd.Timedelta(hours=1)
)

issue_weather = issue_weather[
    [
        "forecast_timestamp_nairobi",
        "ambient_temperature_c",
    ]
].rename(columns={
    "ambient_temperature_c": "ambient_at_forecast_issue_c",
})

# Attach target-hour weather and forecast-issue-time temperature.
data = forecast.merge(
    target_weather,
    on="forecast_timestamp_nairobi",
    how="left",
    validate="one_to_one",
)

data = data.merge(
    issue_weather,
    on="forecast_timestamp_nairobi",
    how="left",
    validate="one_to_one",
)

required_joined_columns = [
    "weather_solar_wh_m2",
    "ambient_target_hour_c",
    "ambient_at_forecast_issue_c",
]

if data[required_joined_columns].isna().any().any():
    raise ValueError(
        "Could not match all forecasts to NASA weather data."
    )

# Ensure that the forecast file and weather file use matching
# observed solar-resource values.
maximum_source_difference = float(
    np.max(
        np.abs(
            data["observed_solar_wh_m2"].to_numpy()
            - data["weather_solar_wh_m2"].to_numpy()
        )
    )
)

if maximum_source_difference > 0.1:
    raise ValueError(
        "Observed solar values in the forecast and weather files "
        f"differ by up to {maximum_source_difference:.3f} Wh/m²."
    )

# ------------------------------------------------------------
# 3. DEFINE THE ILLUSTRATIVE LOAD, TARIFF, PV AND BATTERY
# ------------------------------------------------------------

# Illustrative 24-hour demand profile from the earlier simulation.
# This same profile is repeated each day; it is not measured
# Nairobi electricity-demand data.
demand_profile_kw = np.array([
    2.0, 1.8, 1.7, 1.7,
    1.8, 2.2, 3.5, 5.0,
    4.5, 3.8, 3.5, 3.7,
    4.2, 4.0, 3.8, 4.0,
    4.5, 5.5, 7.0, 8.0,
    7.5, 6.0, 4.0, 2.8,
], dtype=float)

if len(demand_profile_kw) != 24:
    raise ValueError("The demand profile must have 24 values.")

# Illustrative import tariff in generic currency units (CU) per kWh.
# This is not a Kenyan utility tariff.
import_tariff_profile_cu_per_kwh = np.array(
    [0.10] * 6
    + [0.16] * 4
    + [0.14] * 7
    + [0.30] * 5
    + [0.12] * 2,
    dtype=float,
)

# Illustrative export credit in generic CU per kWh.
export_credit_profile_cu_per_kwh = np.full(24, 0.05)

if len(import_tariff_profile_cu_per_kwh) != 24:
    raise ValueError("The import-price profile must have 24 values.")

if len(export_credit_profile_cu_per_kwh) != 24:
    raise ValueError("The export-credit profile must have 24 values.")

time_step_hours = 1.0

# PV model parameters
pv_rated_power_kw = 10.0
standard_irradiance_w_m2 = 1000.0
noct_c = 45.0
temperature_coefficient_per_c = -0.004
inverter_efficiency = 0.96

# Maximum AC output in this simplified model
maximum_pv_ac_power_kw = (
    pv_rated_power_kw * inverter_efficiency
)

# Battery parameters
battery_capacity_kwh = 20.0
initial_soc_percent = 20.0
minimum_soc_percent = 20.0
maximum_soc_percent = 100.0

maximum_charge_power_kw = 5.0
maximum_discharge_power_kw = 5.0

charging_efficiency = 0.95
discharging_efficiency = 0.95

minimum_battery_energy_kwh = (
    battery_capacity_kwh * minimum_soc_percent / 100.0
)
maximum_battery_energy_kwh = (
    battery_capacity_kwh * maximum_soc_percent / 100.0
)
initial_battery_energy_kwh = (
    battery_capacity_kwh * initial_soc_percent / 100.0
)

# The simple controller discharges only at this example high price.
# The threshold is generic CU/kWh, not a real tariff.
high_price_threshold_cu_per_kwh = 0.30

# ------------------------------------------------------------
# 4. ESTIMATE PV AC POWER FROM HOURLY SOLAR RESOURCE
# ------------------------------------------------------------

def estimate_pv_ac_power_kw(solar_resource_wh_m2, ambient_temperature_c):
    """
    Estimate average AC PV power for each hourly interval.

    NASA supplies solar energy on a horizontal surface in Wh/m².
    This simplified model converts it to an average irradiance,
    estimates cell temperature with a NOCT approximation, applies
    a temperature correction and inverter efficiency, then clips
    output to the assumed inverter limit.
    """
    solar_resource = np.maximum(
        np.asarray(solar_resource_wh_m2, dtype=float),
        0.0,
    )
    ambient_temperature = np.asarray(
        ambient_temperature_c,
        dtype=float,
    )

    average_irradiance_w_m2 = (
        solar_resource / time_step_hours
    )

    estimated_cell_temperature_c = (
        ambient_temperature
        + ((noct_c - 20.0) / 800.0)
        * average_irradiance_w_m2
    )

    temperature_factor = np.maximum(
        0.0,
        1.0
        + temperature_coefficient_per_c
        * (estimated_cell_temperature_c - 25.0),
    )

    estimated_ac_power_kw = (
        pv_rated_power_kw
        * (average_irradiance_w_m2 / standard_irradiance_w_m2)
        * temperature_factor
        * inverter_efficiency
    )

    return np.clip(
        estimated_ac_power_kw,
        0.0,
        maximum_pv_ac_power_kw,
    )


# Actual modeled PV uses the target-hour NASA solar estimate and
# target-hour ambient temperature. These values are used only to
# simulate the realized outcome, not as a forecast-model input.
actual_pv_power_kw = estimate_pv_ac_power_kw(
    data["observed_solar_wh_m2"].to_numpy(),
    data["ambient_target_hour_c"].to_numpy(),
)

# Forecasted PV uses predicted solar resource and only the
# temperature available when the forecast was issued.
forecast_pv_power_kw = {}

for case_id, prediction_column in prediction_columns.items():
    forecast_pv_power_kw[case_id] = estimate_pv_ac_power_kw(
        data[prediction_column].to_numpy(),
        data["ambient_at_forecast_issue_c"].to_numpy(),
    )

# Repeat the illustrative load and example prices by local hour.
local_hour = (
    data["forecast_timestamp_nairobi"].dt.hour.to_numpy()
)

demand_power_kw = demand_profile_kw[local_hour]

import_price_cu_per_kwh = (
    import_tariff_profile_cu_per_kwh[local_hour]
)
export_credit_cu_per_kwh = (
    export_credit_profile_cu_per_kwh[local_hour]
)

# ------------------------------------------------------------
# 5. CREATE A NO-BATTERY BASELINE
# ------------------------------------------------------------

actual_power_balance_kw = (
    actual_pv_power_kw - demand_power_kw
)

no_battery_import_kw = np.maximum(
    -actual_power_balance_kw,
    0.0,
)
no_battery_export_kw = np.maximum(
    actual_power_balance_kw,
    0.0,
)

number_of_hours = len(data)
zero_power = np.zeros(number_of_hours)

no_battery_results = {
    "grid_import_power_kw": no_battery_import_kw,
    "grid_export_power_kw": no_battery_export_kw,
    "battery_charge_power_kw": zero_power.copy(),
    "battery_discharge_power_kw": zero_power.copy(),
    "battery_energy_kwh": np.full(number_of_hours, np.nan),
    "battery_soc_percent": np.full(number_of_hours, np.nan),
    "initial_soc_percent": np.nan,
    "final_soc_percent": np.nan,
}

# ------------------------------------------------------------
# 6. SIMULATE THE FORECAST-INFORMED BATTERY CONTROLLER
# ------------------------------------------------------------

def simulate_forecast_controller(
    predicted_pv_power_kw,
    realized_pv_power_kw,
):
    """
    Simulate one simple forecast-informed battery policy.

    Policy:
    - If the forecast predicts solar surplus, charge from actual
      solar surplus, up to the forecasted amount and battery limits.
    - If the forecast predicts a deficit during the example
      high-price hours, discharge up to the forecasted deficit and
      battery limits.
    - Do not charge from the grid.
    - Use realized PV and demand to calculate actual grid exchange.
    """
    battery_energy_kwh = initial_battery_energy_kwh

    grid_import = np.zeros(number_of_hours)
    grid_export = np.zeros(number_of_hours)
    charge_power = np.zeros(number_of_hours)
    discharge_power = np.zeros(number_of_hours)
    battery_energy = np.zeros(number_of_hours)
    battery_soc = np.zeros(number_of_hours)

    for index in range(number_of_hours):
        forecast_balance_kw = (
            predicted_pv_power_kw[index]
            - demand_power_kw[index]
        )

        realized_balance_kw = (
            realized_pv_power_kw[index]
            - demand_power_kw[index]
        )

        desired_charge_power_kw = 0.0
        desired_discharge_power_kw = 0.0

        if forecast_balance_kw > 1e-9:
            # The forecast predicts a solar surplus.
            # Limit the requested charge by the battery headroom.
            remaining_capacity_kwh = max(
                0.0,
                maximum_battery_energy_kwh
                - battery_energy_kwh,
            )

            maximum_charge_allowed_by_capacity_kw = (
                remaining_capacity_kwh
                / (charging_efficiency * time_step_hours)
            )

            desired_charge_power_kw = min(
                forecast_balance_kw,
                maximum_charge_power_kw,
                maximum_charge_allowed_by_capacity_kw,
            )

        elif (
            forecast_balance_kw < -1e-9
            and import_price_cu_per_kwh[index]
            >= high_price_threshold_cu_per_kwh - 1e-12
        ):
            # The forecast predicts a deficit during an example
            # high-price hour. Limit discharge by stored energy.
            available_energy_kwh = max(
                0.0,
                battery_energy_kwh
                - minimum_battery_energy_kwh,
            )

            maximum_discharge_allowed_by_energy_kw = (
                available_energy_kwh
                * discharging_efficiency
                / time_step_hours
            )

            desired_discharge_power_kw = min(
                -forecast_balance_kw,
                maximum_discharge_power_kw,
                maximum_discharge_allowed_by_energy_kw,
            )

        # Physical solar surplus or deficit during this interval.
        # This prevents the forecast command from violating the
        # actual simulated power balance.
        actual_surplus_kw = max(realized_balance_kw, 0.0)
        actual_deficit_kw = max(-realized_balance_kw, 0.0)

        # Charge only from actual solar surplus.
        charge_power[index] = min(
            desired_charge_power_kw,
            actual_surplus_kw,
            maximum_charge_power_kw,
        )

        # Discharge only to serve an actual load deficit.
        discharge_power[index] = min(
            desired_discharge_power_kw,
            actual_deficit_kw,
            maximum_discharge_power_kw,
        )

        # Battery energy change includes charge and discharge losses.
        battery_energy_kwh += (
            charge_power[index]
            * time_step_hours
            * charging_efficiency
        )

        battery_energy_kwh -= (
            discharge_power[index]
            * time_step_hours
            / discharging_efficiency
        )

        # Keep the simulated battery within its allowed energy limits.
        battery_energy_kwh = min(
            battery_energy_kwh,
            maximum_battery_energy_kwh,
        )
        battery_energy_kwh = max(
            battery_energy_kwh,
            minimum_battery_energy_kwh,
        )

        # Grid supplies any remaining deficit or accepts remaining surplus.
        grid_import[index] = max(
            actual_deficit_kw - discharge_power[index],
            0.0,
        )

        grid_export[index] = max(
            actual_surplus_kw - charge_power[index],
            0.0,
        )

        battery_energy[index] = battery_energy_kwh
        battery_soc[index] = (
            battery_energy_kwh
            / battery_capacity_kwh
            * 100.0
        )

    return {
        "grid_import_power_kw": grid_import,
        "grid_export_power_kw": grid_export,
        "battery_charge_power_kw": charge_power,
        "battery_discharge_power_kw": discharge_power,
        "battery_energy_kwh": battery_energy,
        "battery_soc_percent": battery_soc,
        "initial_soc_percent": initial_soc_percent,
        "final_soc_percent": float(battery_soc[-1]),
    }


# ------------------------------------------------------------
# 7. RUN EACH FORECAST-BASED CASE
# ------------------------------------------------------------

case_labels = {
    "no_battery": "No battery",
    "persistence": "Persistence forecast",
    "month_hour_average": "Month-hour average forecast",
    "random_forest": "Random Forest forecast",
    "realized_solar_reference": (
        "Realized-solar reference (not forecast)"
    ),
}

case_results = {
    "no_battery": no_battery_results,
}

for case_id in [
    "persistence",
    "month_hour_average",
    "random_forest",
]:
    case_results[case_id] = simulate_forecast_controller(
        predicted_pv_power_kw=forecast_pv_power_kw[case_id],
        realized_pv_power_kw=actual_pv_power_kw,
    )

# This reference uses the realized solar output to make the
# controller's decision. It is an offline comparison, not a forecast.
case_results["realized_solar_reference"] = (
    simulate_forecast_controller(
        predicted_pv_power_kw=actual_pv_power_kw,
        realized_pv_power_kw=actual_pv_power_kw,
    )
)

# ------------------------------------------------------------
# 8. VALIDATE POWER BALANCE AND BATTERY LIMITS
# ------------------------------------------------------------

tolerance = 1e-6
validation_rows = []

for case_id, result in case_results.items():
    power_balance_residual_kw = (
        actual_pv_power_kw
        + result["battery_discharge_power_kw"]
        + result["grid_import_power_kw"]
        - demand_power_kw
        - result["battery_charge_power_kw"]
        - result["grid_export_power_kw"]
    )

    maximum_power_error_kw = float(
        np.max(np.abs(power_balance_residual_kw))
    )

    if maximum_power_error_kw > tolerance:
        raise RuntimeError(
            f"Power-balance check failed for {case_labels[case_id]}."
        )

    if case_id != "no_battery":
        soc = result["battery_soc_percent"]
        charge = result["battery_charge_power_kw"]
        discharge = result["battery_discharge_power_kw"]

        if not np.all(
            (soc >= minimum_soc_percent - tolerance)
            & (soc <= maximum_soc_percent + tolerance)
        ):
            raise RuntimeError(
                f"SOC limits failed for {case_labels[case_id]}."
            )

        if np.any(charge > maximum_charge_power_kw + tolerance):
            raise RuntimeError(
                f"Charge power limit failed for {case_labels[case_id]}."
            )

        if np.any(discharge > maximum_discharge_power_kw + tolerance):
            raise RuntimeError(
                "Discharge power limit failed for "
                f"{case_labels[case_id]}."
            )

        if np.any(
            (charge > tolerance)
            & (discharge > tolerance)
        ):
            raise RuntimeError(
                "Simultaneous charging and discharging occurred for "
                f"{case_labels[case_id]}."
            )

    validation_rows.append({
        "case": case_labels[case_id],
        "maximum_power_balance_error_kw": maximum_power_error_kw,
    })

validation = pd.DataFrame(validation_rows)

# ------------------------------------------------------------
# 9. CALCULATE ANNUAL ENERGY AND COST SUMMARIES
# ------------------------------------------------------------

summary_rows = []

for case_id, result in case_results.items():
    imports_kwh = (
        result["grid_import_power_kw"].sum()
        * time_step_hours
    )
    exports_kwh = (
        result["grid_export_power_kw"].sum()
        * time_step_hours
    )

    import_charge_cu = np.sum(
        result["grid_import_power_kw"]
        * time_step_hours
        * import_price_cu_per_kwh
    )

    export_credit_cu = np.sum(
        result["grid_export_power_kw"]
        * time_step_hours
        * export_credit_cu_per_kwh
    )

    if case_id == "no_battery":
        charge_energy_from_bus_kwh = 0.0
        energy_delivered_from_battery_kwh = 0.0
        final_soc = np.nan
    else:
        charge_energy_from_bus_kwh = (
            result["battery_charge_power_kw"].sum()
            * time_step_hours
        )
        energy_delivered_from_battery_kwh = (
            result["battery_discharge_power_kw"].sum()
            * time_step_hours
        )
        final_soc = result["final_soc_percent"]

    summary_rows.append({
        "case": case_labels[case_id],
        "grid_import_kwh": imports_kwh,
        "grid_export_kwh": exports_kwh,
        "net_grid_import_kwh": imports_kwh - exports_kwh,
        "import_charge_cu": import_charge_cu,
        "export_credit_cu": export_credit_cu,
        "net_grid_cost_cu": import_charge_cu - export_credit_cu,
        "peak_grid_import_kw": float(
            result["grid_import_power_kw"].max()
        ),
        "battery_charge_energy_from_bus_kwh": (
            charge_energy_from_bus_kwh
        ),
        "battery_energy_delivered_to_load_kwh": (
            energy_delivered_from_battery_kwh
        ),
        "initial_battery_soc_percent": (
            result["initial_soc_percent"]
        ),
        "final_battery_soc_percent": final_soc,
        "max_power_balance_error_kw": (
            validation.loc[
                validation["case"] == case_labels[case_id],
                "maximum_power_balance_error_kw",
            ].iloc[0]
        ),
    })

summary = pd.DataFrame(summary_rows)

# ------------------------------------------------------------
# 10. SAVE THE HOURLY RESULTS AND SUMMARY
# ------------------------------------------------------------

hourly_results = pd.DataFrame({
    "timestamp_nairobi": (
        data["forecast_timestamp_nairobi"].astype(str)
    ),
    "local_hour": local_hour,
    "demand_assumed_kw": demand_power_kw,
    "ambient_at_forecast_issue_c": (
        data["ambient_at_forecast_issue_c"].to_numpy()
    ),
    "ambient_target_hour_c": (
        data["ambient_target_hour_c"].to_numpy()
    ),
    "observed_nasa_solar_resource_wh_m2": (
        data["observed_solar_wh_m2"].to_numpy()
    ),
    "modeled_actual_pv_power_kw": actual_pv_power_kw,
    "import_price_cu_per_kwh": import_price_cu_per_kwh,
    "export_credit_cu_per_kwh": export_credit_cu_per_kwh,
})

for case_id, prediction_column in prediction_columns.items():
    hourly_results[
        f"{case_id}_solar_forecast_wh_m2"
    ] = data[prediction_column].to_numpy()

    hourly_results[
        f"{case_id}_modeled_pv_forecast_kw"
    ] = forecast_pv_power_kw[case_id]

for case_id, result in case_results.items():
    hourly_results[
        f"{case_id}_grid_import_kw"
    ] = result["grid_import_power_kw"]

    hourly_results[
        f"{case_id}_grid_export_kw"
    ] = result["grid_export_power_kw"]

    hourly_results[
        f"{case_id}_battery_charge_kw"
    ] = result["battery_charge_power_kw"]

    hourly_results[
        f"{case_id}_battery_discharge_kw"
    ] = result["battery_discharge_power_kw"]

    hourly_results[
        f"{case_id}_battery_energy_kwh"
    ] = result["battery_energy_kwh"]

    hourly_results[
        f"{case_id}_battery_soc_percent"
    ] = result["battery_soc_percent"]

hourly_file = (
    project_folder
    / "nairobi_2023_forecast_informed_dispatch_hourly.csv"
)
summary_file = (
    project_folder
    / "nairobi_2023_forecast_informed_dispatch_summary.csv"
)

hourly_results.to_csv(hourly_file, index=False)
summary.to_csv(summary_file, index=False)

# ------------------------------------------------------------
# 11. CREATE A FIRST-WEEK GRAPH
# ------------------------------------------------------------

hours_to_plot = min(24 * 7, number_of_hours)
plot_times = (
    data["forecast_timestamp_nairobi"].iloc[:hours_to_plot]
)

figure, axes = plt.subplots(
    3,
    1,
    figsize=(14, 12),
    sharex=False,
)

# Panel 1: modeled PV output and the illustrative demand profile
axes[0].plot(
    plot_times,
    actual_pv_power_kw[:hours_to_plot],
    label="Modeled PV output from observed NASA resource",
    color="darkorange",
)
axes[0].plot(
    plot_times,
    demand_power_kw[:hours_to_plot],
    label="Illustrative demand profile",
    color="royalblue",
)
axes[0].set_title(
    "Nairobi Forecast-Informed Dispatch Backtest — First Week"
)
axes[0].set_ylabel("Power (kW)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Panel 2: battery state of charge for each controller
soc_times = [plot_times.iloc[0]] + [
    timestamp + pd.Timedelta(hours=time_step_hours)
    for timestamp in plot_times
]

for case_id in [
    "persistence",
    "month_hour_average",
    "random_forest",
    "realized_solar_reference",
]:
    result = case_results[case_id]

    soc_values = np.concatenate((
        [initial_soc_percent],
        result["battery_soc_percent"][:hours_to_plot],
    ))

    axes[1].plot(
        soc_times,
        soc_values,
        label=case_labels[case_id],
    )

axes[1].axhline(
    minimum_soc_percent,
    color="red",
    linestyle="--",
    label="Minimum SOC",
)
axes[1].set_ylabel("Battery SOC (%)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Panel 3: grid imports for each case
for case_id, result in case_results.items():
    axes[2].plot(
        plot_times,
        result["grid_import_power_kw"][:hours_to_plot],
        label=case_labels[case_id],
        alpha=0.9,
    )

axes[2].set_xlabel("Time (Nairobi local time)")
axes[2].set_ylabel("Grid import (kW)")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

figure.autofmt_xdate()
figure.tight_layout()

first_week_plot_file = (
    project_folder
    / "nairobi_2023_forecast_informed_dispatch_first_week.png"
)
figure.savefig(first_week_plot_file, dpi=150)
plt.close(figure)

# ------------------------------------------------------------
# 12. CREATE A MONTHLY GRID-IMPORT GRAPH
# ------------------------------------------------------------

hourly_results["month_number"] = (
    data["forecast_timestamp_nairobi"].dt.month.to_numpy()
)

monthly_import_columns = [
    f"{case_id}_grid_import_kw"
    for case_id in case_results
]

monthly_imports = (
    hourly_results
    .groupby("month_number")[monthly_import_columns]
    .sum()
    * time_step_hours
)

monthly_imports.columns = [
    case_labels[case_id]
    for case_id in case_results
]

monthly_labels = [
    pd.Timestamp(year=2023, month=month, day=1).strftime("%b")
    for month in range(1, 13)
]

monthly_imports.index = monthly_labels

figure, axis = plt.subplots(figsize=(12, 6))

for case_name in monthly_imports.columns:
    axis.plot(
        monthly_imports.index,
        monthly_imports[case_name],
        marker="o",
        label=case_name,
    )

axis.set_title(
    "Monthly Grid Imports by Dispatch Case — Nairobi, 2023"
)
axis.set_xlabel("Month")
axis.set_ylabel("Grid energy imported (kWh)")
axis.grid(True, alpha=0.3)
axis.legend()
figure.tight_layout()

monthly_plot_file = (
    project_folder
    / "nairobi_2023_forecast_informed_dispatch_monthly_imports.png"
)
figure.savefig(monthly_plot_file, dpi=150)
plt.close(figure)

# ------------------------------------------------------------
# 13. PRINT THE RESULTS AND IMPORTANT LIMITATIONS
# ------------------------------------------------------------

print("\nFORECAST-INFORMED DISPATCH BACKTEST")
print("-----------------------------------")
print(f"Hourly records simulated: {number_of_hours}")
print(f"PV array rating: {pv_rated_power_kw:.1f} kW")
print(f"Battery capacity: {battery_capacity_kwh:.1f} kWh")
print(f"Battery initial SOC: {initial_soc_percent:.1f}%")
print(
    "Controller: charge from realized solar surplus when the "
    "forecast predicts surplus; discharge during the example "
    "high-price hours when the forecast predicts a deficit."
)

print("\nANNUAL CASE SUMMARY")
print("-------------------")
print(
    summary[
        [
            "case",
            "grid_import_kwh",
            "grid_export_kwh",
            "net_grid_cost_cu",
            "peak_grid_import_kw",
            "final_battery_soc_percent",
        ]
    ].round(2).to_string(index=False)
)

print("\nPOWER-BALANCE CHECKS")
print("--------------------")
print(validation.round(8).to_string(index=False))

print("\nSAVED FILES")
print("-----------")
print(f"Hourly dispatch results: {hourly_file.name}")
print(f"Annual case summary: {summary_file.name}")
print(f"First-week graph: {first_week_plot_file.name}")
print(f"Monthly import graph: {monthly_plot_file.name}")

print(
    "\nInterpretation limits: the load profile repeats the same "
    "illustrative day all year; prices are generic CU, not Kenyan "
    "tariffs; PV output is estimated from NASA horizontal solar "
    "resource rather than measured PV production. The realized-solar "
    "reference uses actual target-hour weather and is not a forecast. "
    "Each battery case starts at the same SOC but may end at a "
    "different SOC, so the raw annual costs are not a fair savings "
    "comparison."
)