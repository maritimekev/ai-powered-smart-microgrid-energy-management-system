# AI-Powered Smart Microgrid Energy Management System

An educational Python project exploring solar energy, battery storage, grid exchange, machine-learning forecasting, and forecast-informed microgrid dispatch.

> **Project status:** This simulation prototype includes a supervised Random Forest model that forecasts NASA POWER-estimated solar resource one hour ahead, plus a rule-based battery-dispatch backtest that uses the forecasts. It is not a real-time controller and has not been validated against measurements from a solar installation.

## Project overview

The project contains two related studies:

1. An illustrative 24-hour microgrid simulation with solar PV, electricity demand, battery storage, and grid imports and exports.
2. A Nairobi-focused solar-resource forecasting experiment using NASA POWER data from 2019-2023, followed by a 2023 forecast-informed dispatch backtest.

The Random Forest is the project's machine-learning component. The battery controller uses explicit rules; it is not a learned AI policy. The separate linear-programming script is an optimization benchmark, not an AI model.

## Data source and location

Hourly weather estimates are from [NASA POWER](https://power.larc.nasa.gov/) for an approximate reference point in Nairobi, Kenya (latitude -1.2921, longitude 36.8219).

The analysis uses:

- `ALLSKY_SFC_SW_DWN`: hourly solar-resource estimate for a horizontal surface, in Wh/m^2.
- `T2M`: near-surface air temperature, in degrees Celsius.

The forecasting pipeline uses Nairobi local time and filters the 2019-2023 period to 43,824 hourly records. NASA POWER data are modeled weather estimates, not measurements from a particular solar array or a ground weather station. Horizontal solar resource is not the same as the electrical output of a PV system. No measured Nairobi electricity-demand data are included.

## Machine-learning solar-resource forecast

`forecast_solar_resource_2019_2023.py` trains a Random Forest model to forecast the next hour's NASA-estimated solar resource. It trains on 2019-2022 and evaluates on the full 2023 calendar year using a chronological test set, not a shuffled split.

On the 2023 test set, the forecast errors were:

| Model | Daylight MAE (Wh/m^2) | All-hours MAE (Wh/m^2) |
|---|---:|---:|
| Persistence forecast | 132.702 | 71.733 |
| Month-hour average forecast | 69.852 | 37.437 |
| Random Forest forecast | 21.912 | 11.742 |

The daylight evaluation contains 4,694 hours. The Random Forest's overall 2023 test results were MAE 11.742 Wh/m^2, RMSE 26.005 Wh/m^2, and R^2 0.994. Since solar-resource values are usually near zero at night, daylight errors are useful alongside all-hours metrics.

These scores describe predictions of NASA POWER estimates; they do not establish accuracy against real solar-panel output. The model's `hour_cos` feature has an importance of 0.8865, showing that its predictions are strongly influenced by time of day. The results should not be interpreted as proof that the model can predict cloud events or generalize to other locations.

## Forecast-informed dispatch backtest

`forecast_informed_dispatch.py` simulates 8,760 hours of 2023. It uses a 10 kW PV array, a 20 kWh battery, and the same illustrative 24-hour demand profile repeated each day. The controller uses solar forecasts to guide rule-based charging from modeled solar surplus and discharging during example high-price hours. Modeled PV production and assumed demand are used for the simulated energy balance.

| Case | Grid imports (kWh) | Grid exports (kWh) | Net grid cost (CU) | Peak grid import (kW) |
|---|---:|---:|---:|---:|
| No battery | 22,224.15 | 5,613.51 | 4,576.52 | 8.00 |
| Persistence forecast | 18,461.27 | 1,444.12 | 3,656.13 | 8.00 |
| Month-hour average forecast | 18,262.17 | 1,223.50 | 3,607.43 | 8.00 |
| Random Forest forecast | 18,019.87 | 955.03 | 3,548.16 | 8.00 |
| Realized-solar reference, not a forecast | 17,880.89 | 801.04 | 3,514.17 | 8.00 |

Costs use illustrative time-of-use prices and export credits in generic currency units (CU); they are not Kenyan electricity tariffs. In this simulation, the Random Forest case has a net grid cost about 1.6% lower than the month-hour average case and 3.0% lower than the persistence case. Its cost is about 22.5% lower than the no-battery case, but that comparison includes adding a battery and does not isolate the effect of the Random Forest model.

The realized-solar reference uses the observed target-hour solar resource and is not a forecast that could be used in advance. All battery cases started and ended at 20% state of charge in this run. The battery remained within its simulated energy and power limits, and the hourly power-balance checks passed. Peak grid import remained 8.00 kW in every case, so this controller did not reduce the maximum grid-import peak.

## Original 24-hour example

The original illustrative day produces these results:

| Daily result | No battery | With battery |
|---|---:|---:|
| Grid energy imported | 51.37 kWh | 36.17 kWh |
| Grid energy exported | 22.40 kWh | 5.56 kWh |
| Net grid energy imported | 28.97 kWh | 30.61 kWh |
| Peak grid import | About 7.52 kW | 6.00 kW |

The battery reduces gross grid imports by about 29.6% and lowers the peak import in this example. Net grid energy imported is higher with the battery because charging and discharging have efficiency losses. This original daily comparison does not include electricity prices.

## Installation

From a terminal, clone the repository, enter its folder, and install the Python libraries:

```bash
git clone https://github.com/maritimekev/ai-powered-smart-microgrid-energy-management-system.git
cd ai-powered-smart-microgrid-energy-management-system
python -m pip install numpy pandas matplotlib requests scikit-learn joblib scipy
```

## Run the original 24-hour simulation

Run these commands from the project folder:

```bash
python first_simulation.py
python data_analysis.py
python microgrid_dispatch.py
python performance_analysis.py
python validate_simulation.py
```

Some scripts open graph windows. Close those windows to let the script finish and return to the terminal before running the next command.

## Run the Nairobi forecasting and dispatch workflow

To download the NASA POWER data again, then regenerate the forecast, diagnostics, and dispatch backtest, run:

```bash
python download_nairobi_weather_2019_2023.py
python forecast_solar_resource_2019_2023.py
python analyze_forecast_errors.py
python forecast_informed_dispatch.py
```

The download step requires an internet connection. The scripts and data files are run from the project folder.

## Main project files

- `first_simulation.py` - illustrative solar and demand profiles.
- `data_analysis.py` - hourly calculations and data output.
- `microgrid_dispatch.py` - rule-based battery and grid dispatch.
- `performance_analysis.py` - battery and no-battery comparison.
- `validate_simulation.py` - checks for simulation constraints and power balance.
- `optimized_dispatch.py` - linear-programming dispatch benchmark; optimization, not AI.
- `download_nairobi_weather_2019_2023.py` - downloads hourly NASA POWER estimates.
- `forecast_solar_resource_2019_2023.py` - trains and evaluates the solar-resource Random Forest.
- `analyze_forecast_errors.py` - analyzes forecast errors by time and month.
- `forecast_informed_dispatch.py` - runs the forecast-informed 2023 dispatch backtest.

Generated forecasting and dispatch outputs include:

- `nairobi_nasa_power_2019_2023_hourly.csv`
- `nairobi_solar_forecast_2019_2022_metrics.csv`
- `nairobi_solar_forecast_2019_2022_model.joblib`
- `nairobi_solar_forecast_2023_test_results.csv`
- `nairobi_2023_forecast_informed_dispatch_hourly.csv`
- `nairobi_2023_forecast_informed_dispatch_summary.csv`
- `nairobi_2023_forecast_informed_dispatch_first_week.png`
- `nairobi_2023_forecast_informed_dispatch_monthly_imports.png`

## Limitations and future work

This is a learning and simulation project, not a tool for designing or controlling real electrical equipment. The annual backtest repeats one assumed daily demand profile, uses illustrative prices, and estimates PV output from NASA horizontal solar-resource data. It does not use measured site load or measured PV production.

The model has not been validated against ground-station observations or a real PV installation. The simulations do not include battery degradation or capital costs, and do not represent a complete real-world grid or inverter model. There is no live data feed, hardware control, or ESP32 integration.

Future work could include validation against measured solar and load data, a more detailed PV and battery model, realistic local tariffs and grid limits, improved control comparisons, a dashboard, and optional hardware integration.

## References

- [NASA POWER](https://power.larc.nasa.gov/)
- [NASA POWER hourly point API](https://power.larc.nasa.gov/api/temporal/hourly/point)