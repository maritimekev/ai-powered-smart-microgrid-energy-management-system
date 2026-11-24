import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

# Approximate reference point near central Nairobi
LATITUDE = -1.2921
LONGITUDE = 36.8219

# Request one complete historical year
START_DATE = "20221231"
END_DATE = "20231231"

# NASA POWER hourly parameters:
# ALLSKY_SFC_SW_DWN = all-sky surface shortwave radiation
# T2M = air temperature at 2 metres
query_parameters = {
    "parameters": "ALLSKY_SFC_SW_DWN,T2M",
    "community": "RE",
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": START_DATE,
    "end": END_DATE,
    "format": "JSON",
    "time-standard": "UTC",
}

api_url = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point?"
    + urlencode(query_parameters)
)

request = Request(
    api_url,
    headers={"User-Agent": "AI-Smart-Microgrid-Educational-Project"},
)

print("Requesting hourly weather data from NASA POWER...")
print(f"Reference location: Nairobi ({LATITUDE}, {LONGITUDE})")
print(f"Period: {START_DATE} to {END_DATE}")

# Download and decode the JSON response
try:
    with urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
except HTTPError as error:
    details = error.read().decode("utf-8", errors="replace")
    raise RuntimeError(
        f"NASA POWER returned HTTP error {error.code}:\n"
        f"{details[:1000]}"
    ) from error
except URLError as error:
    raise RuntimeError(
        f"Could not connect to NASA POWER: {error.reason}"
    ) from error

# Extract the hourly parameter data from the response
properties = payload.get("properties", {})
parameter_data = properties.get("parameter")

# Support the alternate data location if the response format differs
if parameter_data is None:
    parameter_data = payload.get("data", {})

if not isinstance(parameter_data, dict):
    raise RuntimeError(
        "Could not find the hourly parameter data in the NASA response."
    )

solar_data = parameter_data.get("ALLSKY_SFC_SW_DWN")
temperature_data = parameter_data.get("T2M")

if not isinstance(solar_data, dict):
    available = list(parameter_data.keys())
    raise RuntimeError(
        "NASA's response did not contain ALLSKY_SFC_SW_DWN. "
        f"Parameters received: {available}"
    )

if not isinstance(temperature_data, dict):
    available = list(parameter_data.keys())
    raise RuntimeError(
        "NASA's response did not contain T2M. "
        f"Parameters received: {available}"
    )

# Keep timestamps that have values for both requested parameters
time_keys = sorted(set(solar_data) & set(temperature_data))

if not time_keys:
    raise RuntimeError(
        "No matching hourly timestamps were returned for both parameters."
    )

# NASA uses a fill value to mark missing data
header = payload.get("header", {})
try:
    fill_value = float(header.get("fill_value", -999.0))
except (TypeError, ValueError):
    fill_value = -999.0


def clean_value(raw_value):
    """Convert an API value to a number; represent missing data as NaN."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return math.nan

    if math.isclose(value, fill_value, rel_tol=0.0, abs_tol=1e-9):
        return math.nan

    return value


# Kenya uses East Africa Time (UTC+3) year-round
nairobi_timezone = timezone(timedelta(hours=3), name="EAT")

records = []

for time_key in time_keys:
    try:
        timestamp_utc = datetime.strptime(
            time_key, "%Y%m%d%H"
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise RuntimeError(
            f"Unexpected timestamp format from NASA POWER: {time_key}"
        ) from error

    timestamp_nairobi = timestamp_utc.astimezone(nairobi_timezone)

    records.append({
        "timestamp_utc": (
            timestamp_utc.isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
        "timestamp_nairobi": timestamp_nairobi.isoformat(
            timespec="seconds"
        ),
        "allsky_sfc_sw_dwn": clean_value(solar_data[time_key]),
        "t2m": clean_value(temperature_data[time_key]),
        "latitude_deg": LATITUDE,
        "longitude_deg": LONGITUDE,
    })

weather_data = pd.DataFrame(records)

# Get units from NASA's response metadata when available
parameter_metadata = payload.get("parameters", {})


def get_units(parameter_name):
    """Return the units NASA reports for a parameter."""
    details = parameter_metadata.get(parameter_name, {})
    if isinstance(details, dict):
        return details.get("units", "not provided in the response")
    return "not provided in the response"


solar_units = get_units("ALLSKY_SFC_SW_DWN")
temperature_units = get_units("T2M")

# Save the downloaded data beside this script
project_folder = Path(__file__).resolve().parent
output_file = project_folder / "nairobi_nasa_power_2023_hourly.csv"
weather_data.to_csv(output_file, index=False)

# The extra UTC date provides boundary hours for Nairobi's
# local calendar year. The analysis will filter to local 2023.
print(
    "\nThe download includes extra UTC boundary hours "
    "for complete Nairobi local-year analysis."
)

missing_solar = weather_data["allsky_sfc_sw_dwn"].isna().sum()
missing_temperature = weather_data["t2m"].isna().sum()

print("\nDOWNLOAD SUMMARY")
print("-----------------")
print(f"Rows received: {len(weather_data)}")
print(f"NASA time standard: {header.get('time_standard', 'not stated')}")
print(f"Solar parameter units: {solar_units}")
print(f"Temperature parameter units: {temperature_units}")
print(f"Missing solar values: {missing_solar}")
print(f"Missing temperature values: {missing_temperature}")

print("\nFirst five records:")
print(weather_data.head().to_string(index=False))

print(f"\nSaved data to: {output_file.name}")
print("Source: NASA POWER API (https://power.larc.nasa.gov/)")