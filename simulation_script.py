import argparse
import configparser
import re
import sys
import os
from typing import Dict

import pandas as pd

sys.path.append(".")

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "Inputs and resources", "config.ini"
)
META_PATH = os.path.join(
    os.path.dirname(__file__), "Inputs and resources", "metadata.ini"
)


def update_config(database: str, demand: str):
    """
    Set config and metadata before initalising
    """

    # Update database path
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    config["paths"]["fp_powerplant_database"] = database
    with open(CONFIG_PATH, "w") as f:
        config.write(f)

    # Update demand path
    meta = configparser.ConfigParser()
    meta.read(META_PATH)
    demand_meta = meta["demand"]["system_electricity_MW"]

    # FIXME: Hack to update metadata
    meta["demand"]["system_electricity_MW"] = re.sub(
        r"(filename=)'.+'", rf"\1'{demand}'", demand_meta
    )
    with open(META_PATH, "w") as f:
        meta.write(f)


def initialise_script():
    from gd_core import PATHS
    from genDispatch import GenFleet, dParameters, set_period, set_scenario

    # FIXME: Set PERIOD as input instead as hardcoded
    # Should calculate based on demand file given
    from Calibration import PERIOD_FROM, PERIOD_TO

    set_period(t_from=PERIOD_FROM, t_to=PERIOD_TO)
    set_scenario({"sys demand": "Baseline"})

    return PATHS["Proj"], GenFleet, dParameters


def pre_calibration(output_path: str, runs: int) -> str:
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

    return pre_calibration_file


def calibration(output_path: str, max_evals: int):
    from Calibration import Calibration, prepare_save_file

    # FIXME: Remove hardcoded paths for 'in_progress' files
    progress_file = os.path.join(
        output_path, "In_Progress", "calibration_in_progress.csv"
    )
    prepare_save_file(progress_file)
    Calibration(max_evals, progress_file)

    return


def run_simulation() -> Dict:
    from genDispatch import solve

    results = solve(simulation_name="Baseline", runchecks=False, daily_progress=False)
    return results


def calc_pp_heat(gen_fleet, parameters, results: Dict) -> None:
    from ToWRF import (
        calc_allheat,
        thermal_analysis,
        calc_heatstreams1,
    )

    calc_allheat(results)
    thermal_analysis(gen_fleet, parameters)

    calc_heatstreams1(
        gen_fleet,
        results,
        by="outlet",
    )
    calc_heatstreams1(
        gen_fleet,
        results,
        by="kind",
        latentheatfactors=parameters["latent heat factors"],
    )


def generate_output(gen_fleet):
    from ToWRF import prep_DUCT_inputs

    prep_DUCT_inputs(
        scenario="baseline",
        GenFleet=gen_fleet,
        PPcells_only=True,
        With_height=True,
        unit="MW",
        day="2016 Apr 15",
        write_to_disk=True,
    )


def main(database: str, demand: str) -> None:
    update_config(database, demand)
    paths, gen_fleet, parameters = initialise_script()
    output_path = os.path.join(paths, "Scripts")

    # 1. Calibration
    pre_calibration(output_path, 2)
    calibration(output_path, 3)

    # 2. Simulation
    results = run_simulation()

    # 3. Calculate heat
    calc_pp_heat(gen_fleet, parameters, results)

    # 4. Create outputs
    generate_output(gen_fleet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs Power Plant simulation and returns heat values."
    )

    parser.add_argument("database", type=str, help="path to power plant database")
    parser.add_argument("demand", type=str, help="path to demand file")

    args = parser.parse_args()
    main(database=args.database, demand=args.demand)
