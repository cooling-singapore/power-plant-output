import argparse
from datetime import date, datetime
import json
import sys
import os
from typing import Dict

import pandas as pd

from script_helper import create_mkt_data_for_sim_year

sys.path.append(".")

SCENARIO_NAME = "Baseline"


def initialise_script(demand: str, market_year: date, day: date) -> None:
    # Read user provided demand
    demand_df = pd.read_csv(demand, index_col=0, parse_dates=True)
    demand_year = int(demand_df.index.year[0])
    date_start = demand_df.index[0]
    date_end = demand_df.index[-1]

    if pd.to_datetime(day) not in demand_df.index:
        raise ValueError(f"Could not find {day} in demand.")

    import gd_core

    if gd_core.PPdispatchError is not None:
        raise gd_core.PPdispatchError

    # Update sys demand
    gd_core.dFundamentals["sys demand"].val = demand_df
    gd_core.dFundamentals["sys demand"].time_params["t_start"] = date_start
    gd_core.dFundamentals["sys demand"].time_params["t_end"] = date_end

    # Set market data
    gd_core.dFundamentals = create_mkt_data_for_sim_year(
        gd_core.dFundamentals, market_year.year, demand_year
    )

    # Changing the simulation period
    gd_core.time.update(
        {
            "simulation period": [date_start, date_end],
            "D_index": pd.period_range(date_start, date_end, freq="D"),
            "DxP_index": pd.period_range(
                date_start,
                date_end,
                freq=gd_core.time["DxP_freq"],
            ),
        }
    )

    from genDispatch import set_scenario

    set_scenario({"sys demand": SCENARIO_NAME})


def run_simulation(database: str):
    import gd_core
    from genDispatch import solve

    # Read user provided database
    gd_core.PPdb = gd_core.pp.GenUnit.set_PPdb(
        database,
        readprms={
            "master": "Stations",
            "params": "Plant Parameters",
            "seeds": "Random Seeds",
        },
    )
    gen_fleet = gd_core.reinit_fleet()

    results = solve(
        simulation_name=SCENARIO_NAME, runchecks=False, daily_progress=False
    )

    return gen_fleet, results


def calc_pp_heat(gen_fleet, results: Dict) -> None:
    from gd_core import dParameters
    from ToWRF import (
        calc_allheat,
        thermal_analysis,
        calc_heatstreams1,
    )

    calc_allheat(results)
    thermal_analysis(gen_fleet, dParameters)

    calc_heatstreams1(
        gen_fleet,
        results,
        by="outlet",
    )
    calc_heatstreams1(
        gen_fleet,
        results,
        by="kind",
        latentheatfactors=dParameters["latent heat factors"],
    )


def to_geojson(dataframe: pd.DataFrame, unit: str) -> Dict:
    def parse_row(row):
        long = row["PP Longitude"]
        lat = row["PP Latitude"]
        height = row["Height [m]"]

        ah_values = row.filter(regex=r"H\d\d", axis=0).values
        ah_dict = {f"AH_{i}:{unit}": v for i, v in enumerate(ah_values)}

        feature = {
            "type": "Feature",
            "properties": {"height:m": height, **ah_dict},
            "geometry": {"type": "Point", "coordinates": [long, lat]},
        }

        return feature

    features = [parse_row(row) for _, row in dataframe.iterrows()]
    geometries = {
        "type": "FeatureCollection",
        "features": features,
    }

    return geometries


def generate_output(output: str, gen_fleet, day: date, unit="MW") -> None:
    from ToWRF import prep_DUCT_inputs

    os.makedirs(output, exist_ok=True)
    day = day.strftime("%Y %m %d")
    WRF_SH, WRF_LH, _, _ = prep_DUCT_inputs(
        scenario=SCENARIO_NAME,
        GenFleet=gen_fleet,
        PPcells_only=True,
        With_height=True,
        unit=unit,
        day=day,
    )

    with open(os.path.join(output, f"SH.geojson"), "w") as f:
        json.dump(to_geojson(WRF_SH, unit), f, indent=2)
    with open(os.path.join(output, f"LH.geojson"), "w") as f:
        json.dump(to_geojson(WRF_LH, unit), f, indent=2)


def main(database: str, demand: str, market_year: date, day: date, output: str) -> None:
    initialise_script(demand, market_year, day)

    # 1. Simulation
    gen_fleet, results = run_simulation(database)

    # 2. Calculate heat
    calc_pp_heat(gen_fleet, results)

    # 3. Create outputs
    generate_output(output, gen_fleet, day)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs Power Plant simulation and returns heat values for a particular day."
    )

    parser.add_argument("database", type=str, help="path to power plant database")
    parser.add_argument("demand", type=str, help="path to demand file")
    parser.add_argument(
        "market_year",
        type=lambda s: datetime.strptime(s, "%Y").date(),
        help="Year of market data",
    )
    parser.add_argument(
        "day",
        type=lambda s: datetime.strptime(s, "%Y%m%d").date(),
        help="day to simulate (YYYYMMDD)",
    )
    parser.add_argument("--output", type=str, help="path to output", default="./output")

    args = parser.parse_args()
    main(
        database=args.database,
        demand=args.demand,
        market_year=args.market_year,
        day=args.day,
        output=args.output,
    )
