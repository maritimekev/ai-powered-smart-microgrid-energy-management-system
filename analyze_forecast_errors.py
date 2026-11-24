import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Find the saved 2023 test predictions beside this script
project_folder = Path(__file__).resolve().parent
input_file = (
    project_folder / "nairobi_solar_forecast_2023_test_results.csv"
)

if not input_file.exists():
    raise FileNotFoundError(
        f"{input_file.name} was not found. "
        "Run forecast_solar_resource_2019_2023.py first."
    )

# Read the test-period predictions
results = pd.read_csv(input_file)

required_columns = {
    "forecast_timestamp_nairobi",
    "observed_solar_wh_m2",
    "persistence_prediction_wh_m2",
    "month_hour_average_prediction_wh_m2",
    "random_forest_prediction_wh_m2",
}

missing_columns = required_columns - set(results.columns)

if missing_columns:
    raise ValueError(
        f"The forecast results are missing: {sorted(missing_columns)}"
    )

# Parse timestamps and convert them to Nairobi local time
results["forecast_timestamp_nairobi"] = pd.to_datetime(
    results["forecast_timestamp_nairobi"],
    utc=True,
    errors="coerce",
).dt.tz_convert("Africa/Nairobi")

if results["forecast_timestamp_nairobi"].isna().any():
    raise ValueError("Some forecast timestamps could not be parsed.")

# Ensure that observed values and predictions are numeric
value_columns = [
    "observed_solar_wh_m2",
    "persistence_prediction_wh_m2",
    "month_hour_average_prediction_wh_m2",
    "random_forest_prediction_wh_m2",
]

for column in value_columns:
    results[column] = pd.to_numeric(
        results[column],
        errors="coerce",
    )

if results[value_columns].isna().any().any():
    raise ValueError(
        "Missing or non-numeric observed values or predictions were found."
    )

# Sort by timestamp and check that this is a complete hourly year
results = (
    results
    .sort_values("forecast_timestamp_nairobi")
    .reset_index(drop=True)
)

if len(results) != 8760:
    raise ValueError(
        f"Expected 8,760 test records for 2023, found {len(results)}."
    )

if results["forecast_timestamp_nairobi"].duplicated().any():
    raise ValueError("Duplicate forecast timestamps were found.")

if not (results["forecast_timestamp_nairobi"].dt.year == 2023).all():
    raise ValueError("The test results include timestamps outside 2023.")

time_steps = (
    results["forecast_timestamp_nairobi"].diff().dropna()
)

if not (time_steps == pd.Timedelta(hours=1)).all():
    raise ValueError(
        "The test results contain a missing or non-hourly timestamp."
    )

# Add calendar columns for grouping the errors
results["month_number"] = (
    results["forecast_timestamp_nairobi"].dt.month
)
results["forecast_hour"] = (
    results["forecast_timestamp_nairobi"].dt.hour
)

# Define daylight from the observed NASA solar-resource value.
# An observed value above zero is counted as daylight.
results["is_daylight"] = results["observed_solar_wh_m2"] > 0

# Map readable model names to their prediction columns
prediction_columns = {
    "Persistence baseline": "persistence_prediction_wh_m2",
    "Month-hour average baseline": (
        "month_hour_average_prediction_wh_m2"
    ),
    "Random Forest": "random_forest_prediction_wh_m2",
}


def calculate_error_metrics(observed, predicted):
    """Calculate MAE, RMSE, and prediction bias."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(observed) == 0:
        return {
            "hours": 0,
            "mae_wh_m2": np.nan,
            "rmse_wh_m2": np.nan,
            "bias_wh_m2": np.nan,
        }

    error = predicted - observed

    return {
        "hours": len(observed),
        "mae_wh_m2": float(np.mean(np.abs(error))),
        "rmse_wh_m2": float(np.sqrt(np.mean(error ** 2))),
        # Positive bias means overprediction; negative means underprediction.
        "bias_wh_m2": float(np.mean(error)),
    }


# Calculate full-year metrics for every method
annual_records = []
observed_all = results["observed_solar_wh_m2"].to_numpy()
daylight_mask = results["is_daylight"].to_numpy()

for model_name, prediction_column in prediction_columns.items():
    predictions = results[prediction_column].to_numpy()

    all_hour_metrics = calculate_error_metrics(
        observed_all,
        predictions,
    )
    daylight_metrics = calculate_error_metrics(
        observed_all[daylight_mask],
        predictions[daylight_mask],
    )

    annual_records.append({
        "model": model_name,
        "test_hours": all_hour_metrics["hours"],
        "overall_mae_wh_m2": all_hour_metrics["mae_wh_m2"],
        "overall_rmse_wh_m2": all_hour_metrics["rmse_wh_m2"],
        "overall_bias_wh_m2": all_hour_metrics["bias_wh_m2"],
        "daylight_hours": daylight_metrics["hours"],
        "daylight_mae_wh_m2": daylight_metrics["mae_wh_m2"],
        "daylight_rmse_wh_m2": daylight_metrics["rmse_wh_m2"],
        "daylight_bias_wh_m2": daylight_metrics["bias_wh_m2"],
    })

annual_summary = pd.DataFrame(annual_records)

# Calculate error metrics separately for each month
monthly_records = []

for month_number in range(1, 13):
    month_data = results.loc[
        results["month_number"] == month_number
    ].copy()

    month_observed = month_data[
        "observed_solar_wh_m2"
    ].to_numpy()
    month_daylight_mask = month_data["is_daylight"].to_numpy()

    for model_name, prediction_column in prediction_columns.items():
        month_predictions = month_data[
            prediction_column
        ].to_numpy()

        all_hour_metrics = calculate_error_metrics(
            month_observed,
            month_predictions,
        )
        daylight_metrics = calculate_error_metrics(
            month_observed[month_daylight_mask],
            month_predictions[month_daylight_mask],
        )

        monthly_records.append({
            "month_number": month_number,
            "month": calendar.month_name[month_number],
            "model": model_name,
            "test_hours": all_hour_metrics["hours"],
            "daylight_hours": daylight_metrics["hours"],
            "all_hour_mae_wh_m2": all_hour_metrics["mae_wh_m2"],
            "all_hour_rmse_wh_m2": all_hour_metrics["rmse_wh_m2"],
            "all_hour_bias_wh_m2": all_hour_metrics["bias_wh_m2"],
            "daylight_mae_wh_m2": daylight_metrics["mae_wh_m2"],
            "daylight_rmse_wh_m2": daylight_metrics["rmse_wh_m2"],
            "daylight_bias_wh_m2": daylight_metrics["bias_wh_m2"],
        })

monthly_summary = pd.DataFrame(monthly_records)

# Calculate daylight error by local hour of day
hourly_records = []

for hour in range(24):
    hour_data = results.loc[
        (results["forecast_hour"] == hour)
        & results["is_daylight"]
    ]

    hour_observed = hour_data[
        "observed_solar_wh_m2"
    ].to_numpy()

    for model_name, prediction_column in prediction_columns.items():
        hour_predictions = hour_data[
            prediction_column
        ].to_numpy()

        hour_metrics = calculate_error_metrics(
            hour_observed,
            hour_predictions,
        )

        hourly_records.append({
            "forecast_hour_nairobi": hour,
            "model": model_name,
            "daylight_observations": hour_metrics["hours"],
            "daylight_mae_wh_m2": hour_metrics["mae_wh_m2"],
            "daylight_rmse_wh_m2": hour_metrics["rmse_wh_m2"],
            "daylight_bias_wh_m2": hour_metrics["bias_wh_m2"],
        })

hourly_summary = pd.DataFrame(hourly_records)

# Save the error summaries
annual_file = (
    project_folder / "nairobi_solar_forecast_2023_annual_errors.csv"
)
monthly_file = (
    project_folder / "nairobi_solar_forecast_2023_monthly_errors.csv"
)
hourly_file = (
    project_folder / "nairobi_solar_forecast_2023_daylight_hourly_errors.csv"
)

annual_summary.to_csv(annual_file, index=False)
monthly_summary.to_csv(monthly_file, index=False)
hourly_summary.to_csv(hourly_file, index=False)

# Graph 1: compare each model's daylight MAE by month
monthly_plot_data = monthly_summary.pivot(
    index="month_number",
    columns="model",
    values="daylight_mae_wh_m2",
).reindex(range(1, 13))

monthly_plot_data.index = [
    calendar.month_abbr[month_number]
    for month_number in range(1, 13)
]

figure, axis = plt.subplots(figsize=(12, 6))
monthly_plot_data.plot(kind="bar", ax=axis)

axis.set_title("Daylight Forecast MAE by Month — Nairobi, 2023")
axis.set_xlabel("Month")
axis.set_ylabel("Daylight MAE (Wh/m² per hourly interval)")
axis.grid(True, axis="y", alpha=0.3)
axis.legend(title="Forecast method")
figure.tight_layout()

monthly_plot_file = (
    project_folder / "nairobi_solar_forecast_2023_monthly_mae.png"
)
figure.savefig(monthly_plot_file, dpi=150)
plt.close(figure)

# Graph 2: compare daylight forecast MAE by local hour
hourly_plot_data = hourly_summary.pivot(
    index="forecast_hour_nairobi",
    columns="model",
    values="daylight_mae_wh_m2",
).reindex(range(24))

figure, axis = plt.subplots(figsize=(12, 6))
hourly_plot_data.plot(
    marker="o",
    ax=axis,
)

axis.set_title(
    "Daylight Forecast MAE by Nairobi Local Hour — 2023"
)
axis.set_xlabel("Forecast hour (Nairobi local time)")
axis.set_ylabel("Daylight MAE (Wh/m² per hourly interval)")
axis.set_xticks(range(24))
axis.grid(True, alpha=0.3)
axis.legend(title="Forecast method")
figure.tight_layout()

hourly_plot_file = (
    project_folder / "nairobi_solar_forecast_2023_hourly_mae.png"
)
figure.savefig(hourly_plot_file, dpi=150)
plt.close(figure)

# Print full-year metrics and the month-by-month daylight comparison
print("\n2023 HOLDOUT FORECAST ERROR")
print("---------------------------")
print("Daylight means the observed NASA value is greater than zero.")
print("Bias = prediction minus observed value.")
print(annual_summary.round(2).to_string(index=False))

daylight_monthly_table = monthly_summary.pivot(
    index="month",
    columns="model",
    values="daylight_mae_wh_m2",
)

month_order = [
    calendar.month_name[month_number]
    for month_number in range(1, 13)
]
daylight_monthly_table = daylight_monthly_table.reindex(month_order)

print("\nDAYLIGHT MAE BY MONTH")
print("---------------------")
print(daylight_monthly_table.round(2).to_string())

print("\nSAVED FILES")
print("-----------")
print(f"Annual error summary: {annual_file.name}")
print(f"Monthly error summary: {monthly_file.name}")
print(f"Hourly error summary: {hourly_file.name}")
print(f"Monthly graph: {monthly_plot_file.name}")
print(f"Hourly graph: {hourly_plot_file.name}")

print(
    "\nThese diagnostics compare forecasts with NASA POWER estimates. "
    "They are not validation against ground-station solar measurements."
)