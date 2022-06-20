import argparse
from datetime import date, datetime
from shutil import copyfile
import sys
import os

import pandas as pd

sys.path.append(".")


def create_mkt_data_for_sim_year(dFundamentals, data_year_to_use, sim_year):
    df2 = pd.Series()

    # Changing Forex
    dFundamentals["forex"].val = dFundamentals["forex"].val[str(data_year_to_use)]
    dFundamentals["forex"].val.index = dFundamentals["forex"].val.index + pd.DateOffset(
        year=sim_year
    )

    df = dFundamentals["forex"].val
    if dFundamentals["sys demand"].val.index[0] not in df.index:
        data = [dFundamentals["forex"].val[0]]
        df1 = pd.Series(data, index=[dFundamentals["sys demand"].val.index[0]])
        df2 = df.append(df1).sort_index()

        dFundamentals["forex"].val = df2
        dFundamentals["forex"].val.columns = ["forex"]

    dFundamentals["forex"].time_params["t_start"] = dFundamentals["forex"].val.index[0]
    dFundamentals["forex"].time_params["t_end"] = dFundamentals["forex"].val.index[-1]
    dFundamentals["forex"]

    # Changing Fuel Prices
    for item in dFundamentals["fuel prices"]:
        dFundamentals["fuel prices"][item].val = dFundamentals["fuel prices"][item].val[
            str(data_year_to_use)
        ]
        dFundamentals["fuel prices"][item].val.index = dFundamentals["fuel prices"][
            item
        ].val.index + pd.DateOffset(year=sim_year)

        df = dFundamentals["fuel prices"][item].val

        if dFundamentals["sys demand"].val.index[0] not in df.index:
            data = [dFundamentals["fuel prices"][item].val[0]]
            df1 = pd.Series(data, index=[dFundamentals["sys demand"].val.index[0]])
            df2 = df.append(df1).sort_index()

            dFundamentals["fuel prices"][item].val = df2
            dFundamentals["fuel prices"][item].val.columns = [item]
            dFundamentals["fuel prices"][item].val

        dFundamentals["fuel prices"][item].time_params["t_start"] = dFundamentals[
            "fuel prices"
        ][item].val.index[0]
        dFundamentals["fuel prices"][item].time_params["t_end"] = dFundamentals[
            "fuel prices"
        ][item].val.index[-1]

    return dFundamentals


def initialise_script(demand: str, market_year: date):
    from gd_core import PATHS, dFundamentals
    from genDispatch import PPdispatchError
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
    results_path =  save_results(best)
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
