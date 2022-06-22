import pandas as pd


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
