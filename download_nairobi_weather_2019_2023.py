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

# The extra UTC date at the start provides boundary hours for
# selecting complete Nairobi-local calendar years from 2019 to 2023.
START_DATE = "20181231"
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
print(f"UTC request period: {START_DATE} to {END_DATE}")
print("This multi-year request may take a little while.")

# Download and decode the JSON response
try:
    with urlopen(request, timeout=180) as response:
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

# Extract the hourly parameter data from NASA's response
properties = payload.get("properties", {})
parameter_data = properties.get("parameter")

# Support an alternate data location if the response format differs
if parameter_data is None:
    parameter_data = payload.get("data", {})

if not isinstance(parameter_data, dict):
    raise RuntimeError(
        "Could not find hourly parameter data in the NASA response."
    )

solar_data = parameter_data.get("ALLSKY_SFC_SW_DWN")
temperature_data = parameter_data.get("T2M")

if not isinstance(solar_data, dict):
    raise RuntimeError(
        "NASA's response did not contain ALLSKY_SFC_SW_DWN. "
        f"Parameters received: {list(parameter_data.keys())}"
    )

if not isinstance(temperature_data, dict):
    raise RuntimeError(
        "NASA's response did not contain T2M. "
        f"Parameters received: {list(parameter_data.keys())}"
    )

# Keep timestamps for which both requested parameters are available
time_keys = sorted(set(solar_data) & set(temperature_data))

if not time_keys:
    raise RuntimeError(
        "NASA POWER returned no matching hourly timestamps."
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

    if math.isclose(
        value,
        fill_value,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return math.nan

    return value


# Kenya uses East Africa Time (UTC+3)
nairobi_timezone = timezone(timedelta(hours=3), name="EAT")

records = []

for time_key in time_keys:
    try:
        timestamp_utc = datetime.strptime(
            time_key,
            "%Y%m%d%H",
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise RuntimeError(
            f"Unexpected timestamp from NASA POWER: {time_key}"
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

# Save under a new filename; this does not overwrite the 2023-only CSV
project_folder = Path(__file__).resolve().parent
output_file = (
    project_folder / "nairobi_nasa_power_2019_2023_hourly.csv"
)
weather_data.to_csv(output_file, index=False)

# The requested UTC period should contain 43,848 hourly records:
# 2019-2023, including leap year 2020, plus one UTC boundary day.
expected_rows = 43848

print("\nDOWNLOAD SUMMARY")
print("-----------------")
print(f"Rows received: {len(weather_data)}")
print(f"Expected rows for requested UTC dates: {expected_rows}")
print(f"NASA time standard: {header.get('time_standard', 'not stated')}")
print(f"Solar parameter units: {solar_units}")
print(f"Temperature parameter units: {temperature_units}")
print(
    "Missing solar values: "
    f"{weather_data['allsky_sfc_sw_dwn'].isna().sum()}"
)
print(
    "Missing temperature values: "
    f"{weather_data['t2m'].isna().sum()}"
)

if len(weather_data) != expected_rows:
    print(
        "\nNote: The row count differs from the expected count. "
        "We will inspect the returned date range before using it."
    )

if len(weather_data) > 0:
    print("\nFirst record:")
    print(weather_data.head(1).to_string(index=False))

    print("\nLast record:")
    print(weather_data.tail(1).to_string(index=False))

print(f"\nSaved data to: {output_file.name}")
print("Source: NASA POWER API (https://power.larc.nasa.gov/)")