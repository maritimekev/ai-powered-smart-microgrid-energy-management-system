import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Find the downloaded weather data beside this Python file
project_folder = Path(__file__).resolve().parent
input_file = project_folder / "nairobi_nasa_power_2023_hourly.csv"

if not input_file.exists():
    raise FileNotFoundError(
        "nairobi_nasa_power_2023_hourly.csv was not found. "
        "Run download_nairobi_weather.py first."
    )

# Read the NASA POWER download
weather = pd.read_csv(input_file)

required_columns = {
    "timestamp_utc",
    "allsky_sfc_sw_dwn",
    "t2m",
}

missing_columns = required_columns - set(weather.columns)

if missing_columns:
    raise ValueError(
        f"The downloaded CSV is missing columns: "
        f"{sorted(missing_columns)}"
    )

# Parse UTC timestamps and convert them to Nairobi local time
weather["timestamp_utc"] = pd.to_datetime(
    weather["timestamp_utc"],
    utc=True,
    errors="coerce",
)

if weather["timestamp_utc"].isna().any():
    raise ValueError("Some UTC timestamps could not be parsed.")

weather["timestamp_nairobi"] = (
    weather["timestamp_utc"].dt.tz_convert("Africa/Nairobi")
)

# Make sure the measured data columns are numeric
weather["allsky_sfc_sw_dwn"] = pd.to_numeric(
    weather["allsky_sfc_sw_dwn"],
    errors="coerce",
)

weather["t2m"] = pd.to_numeric(
    weather["t2m"],
    errors="coerce",
)

# Select the 2023 calendar year using Nairobi local time
start_nairobi = pd.Timestamp(
    "2023-01-01 00:00:00",
    tz="Africa/Nairobi",
)
end_nairobi = pd.Timestamp(
    "2024-01-01 00:00:00",
    tz="Africa/Nairobi",
)

local_year = weather.loc[
    (weather["timestamp_nairobi"] >= start_nairobi)
    & (weather["timestamp_nairobi"] < end_nairobi)
].copy()

local_year = (
    local_year
    .sort_values("timestamp_nairobi")
    .reset_index(drop=True)
)

# A complete non-leap year contains 8,760 hourly records
if len(local_year) != 8760:
    raise ValueError(
        f"Expected 8,760 Nairobi-local records for 2023, "
        f"but found {len(local_year)}."
    )

# Check that the data starts and ends at the expected local hours
expected_last_hour = end_nairobi - pd.Timedelta(hours=1)

if local_year["timestamp_nairobi"].iloc[0] != start_nairobi:
    raise ValueError(
        "The local 2023 data does not begin at 2023-01-01 00:00."
    )

if local_year["timestamp_nairobi"].iloc[-1] != expected_last_hour:
    raise ValueError(
        "The local 2023 data does not end at 2023-12-31 23:00."
    )

# Check that timestamps are unique and consecutive
if local_year["timestamp_nairobi"].duplicated().any():
    raise ValueError("Duplicate local timestamps were found.")

time_gaps = local_year["timestamp_nairobi"].diff().dropna()

if not (time_gaps == pd.Timedelta(hours=1)).all():
    raise ValueError(
        "There is at least one missing or non-hourly timestamp."
    )

# Check for missing solar or temperature values
solar_missing = local_year["allsky_sfc_sw_dwn"].isna().sum()
temperature_missing = local_year["t2m"].isna().sum()

if solar_missing > 0 or temperature_missing > 0:
    raise ValueError(
        f"Missing values found: solar={solar_missing}, "
        f"temperature={temperature_missing}."
    )

# NASA reports solar radiation in Wh/m² for each hourly interval.
# Add the hourly amounts to calculate each day's solar irradiation.
local_year["local_date"] = (
    local_year["timestamp_nairobi"].dt.strftime("%Y-%m-%d")
)

daily = (
    local_year
    .groupby("local_date", as_index=False)
    .agg(
        daily_surface_shortwave_wh_m2=(
            "allsky_sfc_sw_dwn",
            "sum",
        ),
        daily_mean_temperature_2m_c=(
            "t2m",
            "mean",
        ),
        hourly_records=(
            "allsky_sfc_sw_dwn",
            "size",
        ),
    )
)

# Convert Wh/m² per day to kWh/m² per day
daily["daily_surface_shortwave_kwh_m2"] = (
    daily["daily_surface_shortwave_wh_m2"] / 1000.0
)

if len(daily) != 365:
    raise ValueError(
        f"Expected 365 local dates for 2023, but found {len(daily)}."
    )

if not (daily["hourly_records"] == 24).all():
    raise ValueError(
        "At least one local date does not have exactly 24 hourly records."
    )

# Calculate average daily solar resource for each month
daily["month_number"] = (
    pd.to_datetime(daily["local_date"]).dt.month
)

monthly = (
    daily
    .groupby("month_number", as_index=False)
    .agg(
        mean_daily_surface_shortwave_kwh_m2=(
            "daily_surface_shortwave_kwh_m2",
            "mean",
        ),
        mean_daily_temperature_2m_c=(
            "daily_mean_temperature_2m_c",
            "mean",
        ),
    )
)

monthly["month"] = monthly["month_number"].map(
    lambda month_number: calendar.month_name[month_number]
)

# Calculate annual and daily summary values
annual_surface_shortwave_kwh_m2 = (
    local_year["allsky_sfc_sw_dwn"].sum() / 1000.0
)

mean_daily_surface_shortwave_kwh_m2 = (
    daily["daily_surface_shortwave_kwh_m2"].mean()
)

peak_day_index = daily[
    "daily_surface_shortwave_kwh_m2"
].idxmax()

peak_day = daily.loc[peak_day_index]

# Save the cleaned hourly data and summaries
hourly_output = (
    project_folder / "nairobi_2023_local_hourly.csv"
)
daily_output = (
    project_folder / "nairobi_2023_daily_summary.csv"
)
monthly_output = (
    project_folder / "nairobi_2023_monthly_summary.csv"
)

local_year.to_csv(hourly_output, index=False)
daily.to_csv(daily_output, index=False)
monthly.to_csv(monthly_output, index=False)

# Graph 1: daily solar resource across 2023
daily_dates = pd.to_datetime(daily["local_date"])

figure, axis = plt.subplots(figsize=(12, 5))
axis.plot(
    daily_dates,
    daily["daily_surface_shortwave_kwh_m2"],
    linewidth=0.8,
    color="darkorange",
)
axis.set_title(
    "NASA POWER Estimated Daily Solar Resource — Nairobi, 2023"
)
axis.set_xlabel("Date (Nairobi local time)")
axis.set_ylabel(
    "Daily surface solar irradiation (kWh/m²/day)"
)
axis.grid(True, alpha=0.3)
figure.autofmt_xdate()
figure.tight_layout()

daily_plot = project_folder / "nairobi_daily_solar_resource.png"
figure.savefig(daily_plot, dpi=150)
plt.close(figure)

# Graph 2: average daily solar resource by month
figure, axis = plt.subplots(figsize=(10, 5))
axis.bar(
    monthly["month"],
    monthly["mean_daily_surface_shortwave_kwh_m2"],
    color="darkorange",
)
axis.set_title(
    "Average Daily Solar Resource by Month — Nairobi, 2023"
)
axis.set_xlabel("Month")
axis.set_ylabel(
    "Mean daily surface solar irradiation (kWh/m²/day)"
)
axis.grid(True, axis="y", alpha=0.3)
figure.tight_layout()

monthly_plot = project_folder / "nairobi_monthly_solar_resource.png"
figure.savefig(monthly_plot, dpi=150)
plt.close(figure)

# Print validation and summary information
print("\nNAIROBI WEATHER-DATA CHECK")
print("--------------------------")
print(f"Local-year hourly records: {len(local_year)}")
print(f"Local dates: {len(daily)}")
print(f"First local timestamp: "
      f"{local_year['timestamp_nairobi'].iloc[0].isoformat()}")
print(f"Last local timestamp:  "
      f"{local_year['timestamp_nairobi'].iloc[-1].isoformat()}")
print(f"Missing solar values: {solar_missing}")
print(f"Missing temperature values: {temperature_missing}")

print("\nSOLAR-RESOURCE SUMMARY")
print("----------------------")
print(
    "Annual horizontal-surface solar irradiation: "
    f"{annual_surface_shortwave_kwh_m2:.2f} kWh/m²"
)
print(
    "Mean daily horizontal-surface solar irradiation: "
    f"{mean_daily_surface_shortwave_kwh_m2:.2f} kWh/m²/day"
)
print(
    "Highest-resource day: "
    f"{peak_day['local_date']} "
    f"({peak_day['daily_surface_shortwave_kwh_m2']:.2f} "
    "kWh/m²/day)"
)

print("\nMONTHLY AVERAGES")
print("----------------")
print(
    monthly[
        [
            "month",
            "mean_daily_surface_shortwave_kwh_m2",
            "mean_daily_temperature_2m_c",
        ]
    ].round(2).to_string(index=False)
)

print("\nSAVED FILES")
print("-----------")
print(f"Hourly data: {hourly_output.name}")
print(f"Daily summary: {daily_output.name}")
print(f"Monthly summary: {monthly_output.name}")
print(f"Daily graph: {daily_plot.name}")
print(f"Monthly graph: {monthly_plot.name}")

print(
    "\nThese are NASA-estimated weather and solar-resource data. "
    "This script does not calculate PV output, electricity demand, "
    "battery operation, or AI forecasts."
)