import argparse
import configparser
import json
import re
import sys
import os
from typing import Dict

import pandas as pd

sys.path.append(".")


INPUT_PATH = os.path.join(os.path.dirname(__file__), "Inputs and resources")
CONFIG_PATH = os.path.join(INPUT_PATH, "config.ini")
META_PATH = os.path.join(INPUT_PATH, "metadata.ini")
SCENARIO_NAME = "Baseline"


def update_config(database: str, demand: str):
    """
    Set config and metadata before initalising
    """
    # Get relative path
    _database = os.path.relpath(database, INPUT_PATH)
    _demand = os.path.relpath(demand, os.path.join(INPUT_PATH, "Fundamentals"))

    # Update database path
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    config["paths"]["fp_powerplant_database"] = _database
    with open(CONFIG_PATH, "w") as f:
        config.write(f)

    # Update demand path
    meta = configparser.ConfigParser()
    meta.read(META_PATH)
    demand_meta = meta["demand"]["system_electricity_MW"]

    # FIXME: Hack to update metadata
    meta["demand"]["system_electricity_MW"] = re.sub(
        r"(filename=)'.+'", rf"\1'{_demand}'", demand_meta
    )
    with open(META_PATH, "w") as f:
        meta.write(f)


def initialise_script():
    from gd_core import PATHS, time
    from genDispatch import PPdispatchError
    import Calibration

    if PPdispatchError is not None:
        raise PPdispatchError

    # !! Does not work if demand is from other years
    period = dict(
        [("start", time["common period"][0]), ("end", time["common period"][1])]
    )
    # FIXME: Remove hardcoded value in Calibration
    Calibration.PERIOD_FROM = period["start"]
    Calibration.PERIOD_TO = period["end"]

    return PATHS["Proj"], period


def pre_calibration(output_path: str, runs: int):
    from Calibration import Pre_Calibration, prepare_save_file

    # FIXME: Remove hardcoded paths for 'in_progress' files
    progress_file = os.path.join(
        output_path, "In_Progress", "pre-calibration_in_progress.csv"
    )
    prepare_save_file(progress_file)
    Pre_Calibration(runs)

    # Normalize Parameters
    pre_calibration_file = os.path.join(output_path, "pre-calibration.csv")
    data = pd.read_csv(progress_file, index_col=[0])
    data.to_csv(pre_calibration_file, index=True)


def calibration(output_path: str, max_evals: int):
    from Calibration import Calibration, prepare_save_file

    # FIXME: Remove hardcoded paths for 'in_progress' files
    progress_file = os.path.join(
        output_path, "In_Progress", "calibration_in_progress.csv"
    )

    prepare_save_file(progress_file)
    best, PARAMS_DFS, RANDSEEDS_DFS = Calibration(max_evals, progress_file)

    params = PARAMS_DFS[best - 1].copy()
    seed = RANDSEEDS_DFS[best - 1].copy()

    return params, seed


def run_simulation(params, seed, period):
    from genDispatch import solve, set_scenario, set_period
    from gd_core import PPdb, reinit_fleet

    PPdb["params"] = params
    PPdb["randseeds"] = seed
    gen_fleet = reinit_fleet()

    set_period(period["start"], period["end"])
    set_scenario({"sys demand": SCENARIO_NAME})

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


def generate_output(gen_fleet, period, unit="MW"):
    from ToWRF import prep_DUCT_inputs

    output_path = "output"

    os.makedirs(output_path)
    for _date in pd.date_range(start=period["start"], end=period["end"], freq="D").date:
        day = _date.strftime("%Y %m %d")
        WRF_SH, WRF_LH, _, _ = prep_DUCT_inputs(
            scenario=SCENARIO_NAME,
            GenFleet=gen_fleet,
            PPcells_only=True,
            With_height=True,
            unit=unit,
            day=day,
        )

        day = _date.strftime("%Y%m%d")
        with open(os.path.join(output_path, f"SH_{day}.geojson"), "w") as f:
            json.dump(to_geojson(WRF_SH, unit), f, indent=2)
        with open(os.path.join(output_path, f"LH_{day}.geojson"), "w") as f:
            json.dump(to_geojson(WRF_LH, unit), f, indent=2)


def main(database: str, demand: str) -> None:
    update_config(database, demand)
    paths, period = initialise_script()
    output_path = os.path.join(paths, "Scripts")

    # 1. Calibration
    pre_calibration(output_path, 2)
    params, seed = calibration(output_path, 3)

    # 2. Simulation
    gen_fleet, results = run_simulation(params, seed, period)

    # 3. Calculate heat
    calc_pp_heat(gen_fleet, results)

    # 4. Create outputs
    generate_output(gen_fleet, period)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs Power Plant simulation and returns heat values."
    )

    parser.add_argument("database", type=str, help="path to power plant database")
    parser.add_argument("demand", type=str, help="path to demand file")

    args = parser.parse_args()
    main(database=args.database, demand=args.demand)
