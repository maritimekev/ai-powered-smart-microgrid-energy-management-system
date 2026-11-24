# AI-Powered Smart Microgrid Energy Management System

A Python simulation of a small solar-powered microgrid, including electricity demand, battery storage, and grid imports and exports.

> **Project status:** This is an educational simulation prototype. Its current energy-management controller uses explicit rules. AI forecasting and optimization are planned future extensions and are not implemented yet.

## What the simulation does

The project models an illustrative 24-hour microgrid with:

- A 10 kW solar PV system, with output estimated from hourly solar irradiance and ambient temperature.
- A changing hourly electricity-demand profile.
- A 20 kWh battery with state-of-charge limits, charge/discharge power limits, and efficiency losses.
- Grid imports when solar and the battery cannot meet demand, and grid exports when surplus solar cannot be stored.
- A comparison between grid exchange with and without a battery.

The project produces hourly CSV data and graphs to help inspect the system’s behavior.

## Model assumptions

The current example uses illustrative profiles, not measurements from a real installation.

| Parameter | Value |
|---|---:|
| PV rated capacity | 10 kW |
| Battery capacity | 20 kWh |
| Initial battery state of charge | 20% |
| Minimum / maximum battery SOC | 20% / 100% |
| Maximum charge / discharge power | 5 kW / 5 kW |
| Charge / discharge efficiency | 95% / 95% |
| Simulation interval | 1 hour |

The simulation assumes the grid can supply any remaining shortfall and accept any remaining surplus. Grid power limits, electricity prices, and operating costs are not modeled.

## Example results

For the illustrative profiles currently included in the project:

| Daily result | No battery | With battery |
|---|---:|---:|
| Grid energy imported | 51.37 kWh | 36.17 kWh |
| Grid energy exported | 22.40 kWh | 5.56 kWh |
| Net grid energy imported | 28.97 kWh | 30.61 kWh |
| Peak grid import | about 7.52 kW | 6.00 kW |

The battery reduces gross grid imports by about 29.6% and reduces the peak grid-import power. However, net grid energy imported is about 1.64 kWh higher with the battery because charging and discharging are not perfectly efficient. No electricity prices are included, so these results do not establish whether using the battery saves money.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib

Install the libraries from a terminal:

```bash
python -m pip install numpy pandas matplotlib
```

## Run the simulation

Clone the repository and enter its folder:

```bash
git clone https://github.com/maritimekev/ai-powered-smart-microgrid-energy-management-system.git
cd ai-powered-smart-microgrid-energy-management-system
```

Run the scripts in this order:

```bash
python first_simulation.py
python data_analysis.py
python microgrid_dispatch.py
python performance_analysis.py
```

`first_simulation.py` opens solar and demand graphs. Close the graph windows to return to the terminal before running the next command.

`data_analysis.py` creates `microgrid_hourly_data.csv`. The dispatch and performance scripts use that data and produce further CSV results and graphs.

## Project files

- `first_simulation.py` — solar PV and demand profiles with visualization.
- `data_analysis.py` — NumPy/Pandas calculations and hourly CSV output.
- `microgrid_dispatch.py` — rule-based battery and grid dispatch.
- `performance_analysis.py` — comparison of the battery and no-battery cases.
- `microgrid_hourly_data.csv` — hourly weather, generation, demand, and power-balance data.
- `microgrid_dispatch_results.csv` — hourly battery and grid dispatch results.
- `battery_soc.png`, `grid_power.png`, `performance_comparison.png` — generated result graphs.

## Limitations

This is a learning and simulation project, not a design or control tool for real electrical equipment. The current model uses a single illustrative day and hourly time steps. Its PV and battery models are simplified; it does not include measured site data, detailed battery degradation, grid constraints, electricity tariffs, AI forecasting, or hardware control.

## Roadmap

- Validate the models and add automated tests.
- Test the simulation with suitable measured or publicly available energy data.
- Add solar-generation and demand forecasting, and evaluate forecast accuracy.
- Compare forecast-informed control or optimization against the current rule-based controller.
- Explore an interactive dashboard and, later, optional sensor or microcontroller integration.