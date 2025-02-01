import numpy as np
import pandas as pd
##### Baselines: PrevWeek #####

def prev_week(meter_data: pd.DataFrame):
    """
    Applies the 'PrevWeek' method to obtain peak positions and peak values.
    It is applied to a Data Frame containing the data for a single meter.

    DEPRECATED: It returns a copy of the initial data frame, with two extra columns:
    UPDATED: To maintain consistency with LinearRegression.py, the returned objects are
    only the relevant arrays: one with the peak maximum, one with the peak position

    - "PrevWeek Peak Value": the predicted value via PrevWeek
    - "PrevWeek Peak Position": the predicted value via PrevWeek
    """

    dropped_data = meter_data.drop(columns=["Peak Value", "Peak Position"])

    meter_data["PrevWeek Peak Value"] = dropped_data.apply(np.max, axis=1)  # Axis 1: max per each row
    meter_data["PrevWeek Peak Position"] = dropped_data.apply(np.argmax, axis=1)

    # return meter_data

    peak_value = meter_data["PrevWeek Peak Value"].values
    peak_position = meter_data["PrevWeek Peak Position"].values

    return peak_value, peak_position
