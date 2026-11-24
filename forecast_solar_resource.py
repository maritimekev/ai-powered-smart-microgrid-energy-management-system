from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Find the cleaned Nairobi weather data beside this script
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "nairobi_2023_local_hourly.csv"

if not input_file.exists():
    raise FileNotFoundError(
        "nairobi_2023_local_hourly.csv was not found. "
        "Run analyze_nairobi_weather.py first."
    )

# Load and check the required data
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

# Parse timestamps and order observations chronologically
weather["timestamp_nairobi"] = pd.to_datetime(
    weather["timestamp_nairobi"],
    utc=True,
    errors="coerce",
).dt.tz_convert("Africa/Nairobi")

if weather["timestamp_nairobi"].isna().any():
    raise ValueError("Some Nairobi timestamps could not be parsed.")

weather = (
    weather
    .sort_values("timestamp_nairobi")
    .reset_index(drop=True)
)

# Convert solar-resource and temperature columns to numeric values
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

if len(weather) != 8760:
    raise ValueError(
        f"Expected 8,760 hourly records, but found {len(weather)}."
    )

# Check that timestamps are unique and exactly one hour apart
if weather["timestamp_nairobi"].duplicated().any():
    raise ValueError("Duplicate hourly timestamps were found.")

time_differences = weather["timestamp_nairobi"].diff().dropna()
if not (time_differences == pd.Timedelta(hours=1)).all():
    raise ValueError(
        "The weather data has a missing or non-hourly timestamp."
    )

# Make features using information available at the forecast origin.
# At hour t, the target is the solar-resource value at hour t+1.
for lag in [0, 1, 2, 24, 168]:
    weather[f"solar_lag_{lag}"] = weather["solar_wh_m2"].shift(lag)

weather["temperature_lag_0"] = weather["temperature_c"]
weather["temperature_lag_1"] = weather["temperature_c"].shift(1)

# The forecast time is known in advance, so its calendar features
# can be used without using future weather measurements.
weather["forecast_timestamp"] = (
    weather["timestamp_nairobi"] + pd.Timedelta(hours=1)
)

forecast_hour = weather["forecast_timestamp"].dt.hour
forecast_day_of_year = weather["forecast_timestamp"].dt.day_of_year

weather["hour_sin"] = np.sin(2 * np.pi * forecast_hour / 24)
weather["hour_cos"] = np.cos(2 * np.pi * forecast_hour / 24)
weather["day_of_year_sin"] = np.sin(
    2 * np.pi * forecast_day_of_year / 365.25
)
weather["day_of_year_cos"] = np.cos(
    2 * np.pi * forecast_day_of_year / 365.25
)

# The next hourly NASA solar value is the prediction target
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

# Remove rows that do not have all required lag values or a target
supervised_data = (
    weather[
        feature_columns
        + ["target_solar_wh_m2", "forecast_timestamp"]
    ]
    .dropna()
    .reset_index(drop=True)
)

# Split chronologically: earlier data for training, later data for testing.
# Do not shuffle time-series data.
split_index = int(len(supervised_data) * 0.70)

train_data = supervised_data.iloc[:split_index].copy()
test_data = supervised_data.iloc[split_index:].copy()

if len(train_data) == 0 or len(test_data) == 0:
    raise ValueError("Not enough data for a training and test split.")

X_train = train_data[feature_columns]
y_train = train_data["target_solar_wh_m2"]

X_test = test_data[feature_columns]
y_test = test_data["target_solar_wh_m2"].to_numpy()

# Train a Random Forest regression model
model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=3,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Model predictions are constrained to non-negative solar values
random_forest_prediction = np.maximum(
    model.predict(X_test),
    0.0,
)

# Persistence baseline: predict that the next hour matches this hour
persistence_prediction = np.maximum(
    X_test["solar_lag_0"].to_numpy(),
    0.0,
)

# Report overall and daytime forecast metrics
actual_daylight = y_test > 0

metric_records = []

for model_name, prediction in [
    ("Persistence baseline", persistence_prediction),
    ("Random Forest", random_forest_prediction),
]:
    record = {
        "model": model_name,
        "test_hours": len(y_test),
        "overall_mae_wh_m2": mean_absolute_error(y_test, prediction),
        "overall_rmse_wh_m2": np.sqrt(
            mean_squared_error(y_test, prediction)
        ),
        "overall_r2": r2_score(y_test, prediction),
    }

    if actual_daylight.any():
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
        record["daylight_mae_wh_m2"] = np.nan
        record["daylight_rmse_wh_m2"] = np.nan

    metric_records.append(record)

metrics = pd.DataFrame(metric_records)

# Save timestamped predictions for the held-out test period
test_results = pd.DataFrame({
    "forecast_timestamp_nairobi": (
        test_data["forecast_timestamp"].astype(str).to_numpy()
    ),
    "observed_solar_wh_m2": y_test,
    "persistence_prediction_wh_m2": persistence_prediction,
    "random_forest_prediction_wh_m2": random_forest_prediction,
})

# Save the model evaluation outputs
metrics_file = project_folder / "nairobi_solar_forecast_metrics.csv"
predictions_file = (
    project_folder / "nairobi_solar_forecast_test_results.csv"
)
model_file = project_folder / "nairobi_solar_forecast_model.joblib"

metrics.to_csv(metrics_file, index=False)
test_results.to_csv(predictions_file, index=False)
joblib.dump(model, model_file)

# Plot the first seven days of test predictions
hours_to_plot = min(24 * 7, len(test_results))
plot_hours = np.arange(hours_to_plot)

figure, axis = plt.subplots(figsize=(12, 5))
axis.plot(
    plot_hours,
    test_results["observed_solar_wh_m2"].iloc[:hours_to_plot],
    label="Observed NASA solar resource",
    linewidth=1.5,
)
axis.plot(
    plot_hours,
    test_results["persistence_prediction_wh_m2"].iloc[:hours_to_plot],
    label="Persistence baseline",
    alpha=0.8,
)
axis.plot(
    plot_hours,
    test_results["random_forest_prediction_wh_m2"].iloc[:hours_to_plot],
    label="Random Forest",
    alpha=0.8,
)
axis.set_title(
    "One-Hour-Ahead Nairobi Solar-Resource Forecast "
    "(First Test Week)"
)
axis.set_xlabel("Hour in the test-period week")
axis.set_ylabel("Hourly solar resource (Wh/m²)")
axis.grid(True, alpha=0.3)
axis.legend()
figure.tight_layout()

plot_file = (
    project_folder / "nairobi_solar_forecast_first_test_week.png"
)
figure.savefig(plot_file, dpi=150)
plt.close(figure)

# Show results in the Terminal
print("\nONE-HOUR-AHEAD SOLAR-RESOURCE FORECAST")
print("--------------------------------------")
print("Target: NASA-estimated hourly solar resource (Wh/m²)")
print(f"Usable supervised records: {len(supervised_data)}")
print(f"Training records: {len(train_data)}")
print(f"Test records: {len(test_data)}")
print(
    "Test period: "
    f"{test_data['forecast_timestamp'].iloc[0].isoformat()} "
    "to "
    f"{test_data['forecast_timestamp'].iloc[-1].isoformat()}"
)
print(
    "Evaluation method: chronological split; "
    "one-hour-ahead rolling test."
)

print("\nFORECAST METRICS")
print("----------------")
print(metrics.round(3).to_string(index=False))

print("\nMOST IMPORTANT RANDOM FOREST FEATURES")
print("-------------------------------------")
feature_importance = (
    pd.Series(model.feature_importances_, index=feature_columns)
    .sort_values(ascending=False)
)
print(feature_importance.head(8).round(4).to_string())

print("\nSAVED FILES")
print("-----------")
print(f"Metrics: {metrics_file.name}")
print(f"Test predictions: {predictions_file.name}")
print(f"Trained model: {model_file.name}")
print(f"Test-week graph: {plot_file.name}")

print(
    "\nInterpretation: lower MAE and RMSE mean smaller forecast "
    "errors. If Random Forest does not outperform persistence, "
    "that is a valid result to investigate—not something to hide."
)