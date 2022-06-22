import argparse
from datetime import date, datetime
from shutil import copyfile
import sys
import os

import pandas as pd

from script_helper import create_mkt_data_for_sim_year

sys.path.append(".")


def initialise_script(demand: str, market_year: date):
    from gd_core import PATHS, dFundamentals, PPdispatchError
    import Calibration

    if PPdispatchError is not None:
        raise PPdispatchError

    # Read user provided demand
    demand_df = pd.read_csv(demand, index_col=0, parse_dates=True)
    demand_year = int(demand_df.index.year[0])

    # Update sys demand
    dFundamentals["sys demand"].val = demand_df
    dFundamentals["sys demand"].time_params["t_start"] = demand_df.index[0]
    dFundamentals["sys demand"].time_params["t_end"] = demand_df.index[-1]

    # Set market data
    dFundamentals = create_mkt_data_for_sim_year(
        dFundamentals, market_year.year, demand_year
    )

    # Set calibration year match demand
    # FIXME: Remove hardcoded period value in Calibration
    Calibration.PERIOD_FROM = f"{demand_year} Jan 01"
    Calibration.PERIOD_TO = f"{demand_year} Dec 31"

    return PATHS["Proj"]


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
    best, _, _ = Calibration(max_evals, progress_file)

    return best


def main(demand: str, market_year: date, output: str) -> None:
    # update_config(demand)
    project_path = initialise_script(demand, market_year)

    # Calibration
    output_path = os.path.join(project_path, "Scripts")
    pre_calibration(output_path, 2)
    best = calibration(output_path, 3)

    from Calibration import save_results

    results_path = save_results(best)
    # Copy database to specified path
    copyfile(os.path.join(results_path, "Calibrated_database.xlsx"), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates power plant database")

    parser.add_argument("demand", type=str, help="path to demand file")
    parser.add_argument(
        "market_year",
        type=lambda s: datetime.strptime(s, "%Y").date(),
        help="Year of market data",
    )
    parser.add_argument(
        "--output", type=str, help="path to output", default="./database.xlsx"
    )

    args = parser.parse_args()
    main(demand=args.demand, market_year=args.market_year, output=args.output)
