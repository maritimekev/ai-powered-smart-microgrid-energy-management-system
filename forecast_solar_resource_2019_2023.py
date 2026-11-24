from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Find the multi-year NASA POWER data beside this script
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "nairobi_nasa_power_2019_2023_hourly.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"{input_file.name} was not found. "
        "Run download_nairobi_weather_2019_2023.py first."
    )

# Read the downloaded data
weather = pd.read_csv(input_file)

required_columns = {
    "timestamp_nairobi",
    "allsky_sfc_sw_dwn",
    "t2m",
}
missing_columns = required_columns - set(weather.columns)

if missing_columns:
    raise ValueError(
        f"The weather CSV is missing: {sorted(missing_columns)}"
    )

# Parse timestamps, then ensure they use Nairobi local time
weather["timestamp_nairobi"] = pd.to_datetime(
    weather["timestamp_nairobi"],
    utc=True,
    errors="coerce",
).dt.tz_convert("Africa/Nairobi")

if weather["timestamp_nairobi"].isna().any():
    raise ValueError("Some Nairobi timestamps could not be parsed.")

# Convert the weather measurements to numbers
weather["solar_wh_m2"] = pd.to_numeric(
    weather["allsky_sfc_sw_dwn"],
    errors="coerce",
)
weather["temperature_c"] = pd.to_numeric(
    weather["t2m"],
    errors="coerce",
)

if weather[["solar_wh_m2", "temperature_c"]].isna().any().any():
    raise ValueError("Missing or non-numeric weather values were found.")

# Keep complete Nairobi-local calendar years 2019 through 2023.
# This is 1,826 days because 2020 is a leap year:
# 1,826 * 24 = 43,824 hourly records.
start_nairobi = pd.Timestamp(
    "2019-01-01 00:00:00",
    tz="Africa/Nairobi",
)
end_nairobi = pd.Timestamp(
    "2024-01-01 00:00:00",
    tz="Africa/Nairobi",
)

weather = weather.loc[
    (weather["timestamp_nairobi"] >= start_nairobi)
    & (weather["timestamp_nairobi"] < end_nairobi)
].copy()

weather = (
    weather
    .sort_values("timestamp_nairobi")
    .reset_index(drop=True)
)

if len(weather) != 43824:
    raise ValueError(
        f"Expected 43,824 hourly records for Nairobi-local "
        f"2019–2023, but found {len(weather)}."
    )

# Make sure there are no duplicate or missing hours
if weather["timestamp_nairobi"].duplicated().any():
    raise ValueError("Duplicate hourly timestamps were found.")

time_gaps = weather["timestamp_nairobi"].diff().dropna()

if not (time_gaps == pd.Timedelta(hours=1)).all():
    raise ValueError(
        "The data contains a missing or non-hourly timestamp."
    )

# Create features available at forecast time t.
# The target is the solar-resource value at time t + 1 hour.
for lag in [0, 1, 2, 24, 168]:
    weather[f"solar_lag_{lag}"] = weather["solar_wh_m2"].shift(lag)

weather["temperature_lag_0"] = weather["temperature_c"]
weather["temperature_lag_1"] = weather["temperature_c"].shift(1)

# Calendar information for the hour being forecast is known in advance
weather["forecast_timestamp"] = (
    weather["timestamp_nairobi"] + pd.Timedelta(hours=1)
)

forecast_hour = weather["forecast_timestamp"].dt.hour
forecast_day_of_year = (
    weather["forecast_timestamp"].dt.day_of_year
)

weather["hour_sin"] = np.sin(2 * np.pi * forecast_hour / 24)
weather["hour_cos"] = np.cos(2 * np.pi * forecast_hour / 24)
weather["day_of_year_sin"] = np.sin(
    2 * np.pi * forecast_day_of_year / 365.25
)
weather["day_of_year_cos"] = np.cos(
    2 * np.pi * forecast_day_of_year / 365.25
)

# The target is NASA-estimated solar energy for the next hourly interval
weather["target_solar_wh_m2"] = weather["solar_wh_m2"].shift(-1)

feature_columns = [
    "solar_lag_0",
    "solar_lag_1",
    "solar_lag_2",
    "solar_lag_24",
    "solar_lag_168",
    "temperature_lag_0",
    "temperature_lag_1",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

# Include month and hour for the climatology baseline.
# Drop rows without enough previous data or a next-hour target.
columns_to_keep = feature_columns + [
    "target_solar_wh_m2",
    "forecast_timestamp",
]

supervised_data = (
    weather[columns_to_keep]
    .dropna()
    .reset_index(drop=True)
)

supervised_data["forecast_month"] = (
    supervised_data["forecast_timestamp"].dt.month
)
supervised_data["forecast_hour"] = (
    supervised_data["forecast_timestamp"].dt.hour
)

# Split by forecast timestamp—not randomly:
# train on 2019–2022 and test on 2023.
forecast_year = supervised_data["forecast_timestamp"].dt.year

train_data = supervised_data.loc[
    forecast_year < 2023
].copy()

test_data = supervised_data.loc[
    forecast_year == 2023
].copy()

train_years = sorted(
    train_data["forecast_timestamp"].dt.year.unique().tolist()
)

if train_years != [2019, 2020, 2021, 2022]:
    raise ValueError(
        f"Expected training years 2019–2022, found {train_years}."
    )

if len(test_data) != 8760:
    raise ValueError(
        f"Expected 8,760 test hours in 2023, "
        f"but found {len(test_data)}."
    )

X_train = train_data[feature_columns]
y_train = train_data["target_solar_wh_m2"]

X_test = test_data[feature_columns]
y_test = test_data["target_solar_wh_m2"].to_numpy()

# Train the Random Forest using only 2019–2022 data
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=18,
    min_samples_leaf=10,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Baseline 1: predict that next hour will match the current hour
persistence_prediction = np.maximum(
    X_test["solar_lag_0"].to_numpy(),
    0.0,
)

# Baseline 2: training-set mean for the forecast month and hour
month_hour_average = train_data.groupby(
    ["forecast_month", "forecast_hour"]
)["target_solar_wh_m2"].mean()

test_month_hour_keys = pd.MultiIndex.from_frame(
    test_data[["forecast_month", "forecast_hour"]]
)

month_hour_prediction = month_hour_average.reindex(
    test_month_hour_keys
).to_numpy(dtype=float)

if not np.isfinite(month_hour_prediction).all():
    raise ValueError(
        "Could not create a training-data month/hour baseline."
    )

# Random Forest estimates cannot represent negative solar energy
random_forest_prediction = np.maximum(
    model.predict(X_test),
    0.0,
)

# Compare all three methods on exactly the same 2023 test hours
actual_daylight = y_test > 0

prediction_sets = [
    ("Persistence baseline", persistence_prediction),
    ("Month-hour average baseline", month_hour_prediction),
    ("Random Forest", random_forest_prediction),
]

metric_records = []

for model_name, prediction in prediction_sets:
    record = {
        "model": model_name,
        "test_hours": len(y_test),
        "overall_mae_wh_m2": mean_absolute_error(
            y_test,
            prediction,
        ),
        "overall_rmse_wh_m2": np.sqrt(
            mean_squared_error(y_test, prediction)
        ),
        "overall_r2": r2_score(y_test, prediction),
    }

    if actual_daylight.any():
        record["daylight_hours"] = int(actual_daylight.sum())
        record["daylight_mae_wh_m2"] = mean_absolute_error(
            y_test[actual_daylight],
            prediction[actual_daylight],
        )
        record["daylight_rmse_wh_m2"] = np.sqrt(
            mean_squared_error(
                y_test[actual_daylight],
                prediction[actual_daylight],
            )
        )
    else:
        record["daylight_hours"] = 0
        record["daylight_mae_wh_m2"] = np.nan
        record["daylight_rmse_wh_m2"] = np.nan

    metric_records.append(record)

metrics = pd.DataFrame(metric_records)

# Save timestamped 2023 test predictions
test_results = pd.DataFrame({
    "forecast_timestamp_nairobi": (
        test_data["forecast_timestamp"].astype(str).to_numpy()
    ),
    "observed_solar_wh_m2": y_test,
    "persistence_prediction_wh_m2": persistence_prediction,
    "month_hour_average_prediction_wh_m2": month_hour_prediction,
    "random_forest_prediction_wh_m2": random_forest_prediction,
})

# Save the model with information needed to interpret its features
model_package = {
    "model": model,
    "feature_columns": feature_columns,
    "target": "next-hour NASA POWER solar resource",
    "target_units": "Wh/m^2 per hourly interval",
    "location": {
        "name": "Nairobi, Kenya",
        "latitude": -1.2921,
        "longitude": 36.8219,
    },
    "training_years": [2019, 2020, 2021, 2022],
    "test_year": 2023,
    "forecast_horizon_hours": 1,
}

# Output filenames are different from the earlier 2023-only experiment
metrics_file = (
    project_folder
    / "nairobi_solar_forecast_2019_2022_metrics.csv"
)
predictions_file = (
    project_folder
    / "nairobi_solar_forecast_2023_test_results.csv"
)
model_file = (
    project_folder
    / "nairobi_solar_forecast_2019_2022_model.joblib"
)

metrics.to_csv(metrics_file, index=False)
test_results.to_csv(predictions_file, index=False)
joblib.dump(model_package, model_file)

# Plot the first seven days of the 2023 test period
hours_to_plot = min(24 * 7, len(test_results))
plot_times = test_data["forecast_timestamp"].iloc[:hours_to_plot]

figure, axis = plt.subplots(figsize=(13, 6))

axis.plot(
    plot_times,
    test_results["observed_solar_wh_m2"].iloc[:hours_to_plot],
    label="Observed NASA solar resource",
    linewidth=1.5,
)

axis.plot(
    plot_times,
    test_results["persistence_prediction_wh_m2"].iloc[:hours_to_plot],
    label="Persistence baseline",
    alpha=0.8,
)

axis.plot(
    plot_times,
    test_results[
        "month_hour_average_prediction_wh_m2"
    ].iloc[:hours_to_plot],
    label="Training month-hour average",
    alpha=0.8,
)

axis.plot(
    plot_times,
    test_results["random_forest_prediction_wh_m2"].iloc[:hours_to_plot],
    label="Random Forest",
    alpha=0.8,
)

axis.set_title(
    "One-Hour-Ahead Nairobi Solar-Resource Forecast "
    "(First Week of 2023 Test)"
)
axis.set_xlabel("Forecast time (Nairobi local time)")
axis.set_ylabel("Hourly solar resource (Wh/m²)")
axis.grid(True, alpha=0.3)
axis.legend()
figure.autofmt_xdate()
figure.tight_layout()

plot_file = (
    project_folder
    / "nairobi_solar_forecast_2023_first_week.png"
)
figure.savefig(plot_file, dpi=150)
plt.close(figure)

# Display the experiment setup and results
print("\nNAIROBI SOLAR-RESOURCE FORECAST")
print("--------------------------------")
print("Target: next-hour NASA-estimated solar resource (Wh/m²)")
print(f"Training years: {train_years}")
print("Test year: 2023")
print(f"Training records: {len(train_data)}")
print(f"Test records: {len(test_data)}")
print(
    "Test period: "
    f"{test_data['forecast_timestamp'].iloc[0].isoformat()} "
    "to "
    f"{test_data['forecast_timestamp'].iloc[-1].isoformat()}"
)
print(
    "Evaluation: rolling one-hour-ahead predictions; "
    "test data is not shuffled."
)

print("\nFORECAST METRICS")
print("----------------")
print(metrics.round(3).to_string(index=False))

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print("--------------------------------")
feature_importance = (
    pd.Series(model.feature_importances_, index=feature_columns)
    .sort_values(ascending=False)
)
print(feature_importance.head(8).round(4).to_string())

print("\nSAVED FILES")
print("-----------")
print(f"Metrics: {metrics_file.name}")
print(f"2023 predictions: {predictions_file.name}")
print(f"Trained model: {model_file.name}")
print(f"Forecast graph: {plot_file.name}")

print(
    "\nInterpretation: lower MAE and RMSE mean smaller errors. "
    "Compare the Random Forest with both baselines. This experiment "
    "forecasts NASA solar resource—not PV output, electricity demand, "
    "or battery behavior."
)