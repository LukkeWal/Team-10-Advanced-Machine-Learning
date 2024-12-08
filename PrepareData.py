import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

"""
Prepares the data from https://datadryad.org/stash/dataset/doi:10.5061/dryad.m0cfxpp2c (22-11-2024) 
to be used for machine learning purposes. It requires the data from the before mentioned link to be 
unzipped, and the folder which contains the csv files should be placed in the same folder as this file
"""

def string_to_datetime(time: str) -> datetime:
    """
    Helper function to transform strings of dates found in the raw data to datetime objects
    """
    # Define the format of the input string
    #date_format = "%d-%b-%Y %H:%M:%S"
    date_format = "%d-%b-%Y %H:%M:%S"
    # Convert the string to a datetime object
    try:
        time = datetime.strptime(time, date_format)
    except ValueError:
        # exception for dates with 2 digit year instead of 4 digit year and no time
        time = datetime.strptime(time.split(" ")[0], "%d-%b-%y")
    return time

def read_data(directory="LADPU") -> list[pd.DataFrame]:
    """
    Read the data from the folder and store it in pandas dataframes
    """
    result = []
    total_file_num = len(os.listdir(directory))
    current_file_num = 0
    for filename in os.listdir(directory):
        current_file_num += 1
        print (f"reading data... {current_file_num}/{total_file_num}", end="\r")
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
        data = pd.read_csv(filepath)
        result.append(data)
    print (f"reading data... DONE")
    return result

def load_precomputed_data(filename="time_series_matrix.csv") -> pd.DataFrame:
    """
    Load the data from the precomputed csv file if available
    """
    data = pd.read_csv(filename)
    # read csv dumps everything into strings, we convert it back to numeric/datetime values that we can work with
    # we starty by turning the column names into datetime objects
    data.columns = pd.to_datetime(data.columns)
    # then we convert the cells back into tubple with a float and an int
    for date in data:
        date_data = data[date]
        for i in range(len(date_data)):
            string = data[date].iloc[i]
            string = string.strip("(").strip(")").split(", ")
            string[0] = float(string[0])
            string[1] = int(string[1])
            data[date].iloc[i] = (string[0], string[1])
    return data

def change_interval_to_one_day(data: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Transform the intervals from 15 minutes to 1-day intervals using Pandas groupby.
    """
    result = []
    total_meter_num = len(data)
    current_meter_num = 0
    for meter_data in data:
        current_meter_num += 1
        print(f"Summing measurements to 1 day intervals... {current_meter_num}/{total_meter_num}", end="\r")
        if len(meter_data) <= 1:  # Skip empty meters
            continue
        
        # Ensure INTERVAL_TIME is a datetime object
        meter_data["INTERVAL_TIME"] = meter_data["INTERVAL_TIME"].apply(string_to_datetime)
        meter_data["DATE"] = meter_data["INTERVAL_TIME"].dt.date  # Extract date

        # Aggregate by DATE: sum measurements and count occurrences
        daily_data = meter_data.groupby("DATE").agg(
            SUM_MEASUREMENTS=("INTERVAL_READ", "sum"),
            AMOUNT_OF_MEASUREMENTS=("INTERVAL_READ", "count")
        ).reset_index()
        result.append(daily_data)
    print(f"Summing measurements to 1 day intervals... DONE")
    return result

def format_data(data: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all meters into a single DataFrame where rows represent meters and columns represent dates.
    """
    # Get the full date range
    all_dates = pd.date_range(
        min(meter_data["DATE"].min() for meter_data in data),
        max(meter_data["DATE"].max() for meter_data in data)
    )

    # Process each meter to ensure data is aligned with the full date range
    formatted_data = []
    for meter_data in data:
        # Reindex the data to include all dates, filling missing values with (0, 0) for both columns
        meter_data = meter_data.set_index("DATE").reindex(all_dates, fill_value=0).reset_index()
        
        # Now, reassign the correct column names
        meter_data.columns = ["DATE", "SUM_MEASUREMENTS", "AMOUNT_OF_MEASUREMENTS"]

        # Fill missing values for both columns
        meter_data["SUM_MEASUREMENTS"].fillna(0, inplace=True)
        meter_data["AMOUNT_OF_MEASUREMENTS"].fillna(0, inplace=True)
        
        # Append to formatted_data list
        #print(meter_data[["SUM_MEASUREMENTS", "AMOUNT_OF_MEASUREMENTS"]].values)
        formatted_data.append(list(zip(meter_data["SUM_MEASUREMENTS"], meter_data["AMOUNT_OF_MEASUREMENTS"])))

    # Combine all meters into a single DataFrame
    result = pd.DataFrame(formatted_data, columns=all_dates)

    print(f"Formatting matrix... DONE")
    return result

def remove_first_year():
    """
    TODO: Remove the first year of data since it has missing values
    """
    return

def apply_sliding_window():
    """
    TODO: Turn the time series into shorter time series that can be used as a feature
    """
    return

def z_normalize( dataframe ):

    """
    # normalize the dataframe

    """

    normalizedDf = StandardScaler().fit_transform(dataframe)

    return normalizedDf

def split_train_and_test( dataframe, size=0.8):

    nRows, nCols = dataframe.shape
    trainSize = int( nRows  * size)
    trainData =  dataframe.iloc[:trainSize,:]
    testData = dataframe.iloc[trainSize:,:]


    return trainData, testData

def load_and_save_raw_data(data_dirname = "LADPU", save_filename="time_series_matrix.csv") -> pd.DataFrame:
    """
    loads the data from the original data files
    this is slow so the result is stored for future use
    """
    data = read_data(directory=data_dirname) # read the data from the original files
    data = change_interval_to_one_day(data) # sums all intervals in a day and discards some unneeded columns
    data = format_data(data) # combines all seperate dataframes (1 for each meter) into a single dataframe
    data.to_csv(save_filename, index=False) # save results becaues its a lot of data and takes long
    return data

# --- Andrei's implementation of the sliding window & prev week

def apply_sliding_window(meter_data):
    """
    Given the data **for a single meter**, this function constructs another DataFrame whose colummns are
    
    - the 7 days: week on which we predict
    - (peak value, peak position): belonging to the next week

    """

    length_window_data = 7    # Lenght of window which makes the covariates
    length_window_peak = 7    # Lenght of window from which we extract peaks
    len_series = len(meter_data)

    # Data frame of the desired format
    colnames = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", 
    "Peak Value", "Peak Position"]
    feature_matrix_df = pd.DataFrame(columns = colnames)

    for i in range(len_series - length_window_peak- length_window_data + 1):
        # Separate the two relevant windows: one to save and one to get data from
        week_data = meter_data[i:(i+length_window_data)]
        week_peak = meter_data[(i+length_window_data):(i+length_window_data+length_window_peak)]

        # Find the peak, append data
        peak_value = np.max(week_peak)
        peak_position = np.argmax(week_peak)
        temp = np.array([peak_value, peak_position])

        # Append to the data frame
        feature = np.concatenate((week_data, temp))
        feature_matrix_df.loc[i] = feature
    
    return feature_matrix_df

def construct_feature_matrix_DataFrame(data: pd.DataFrame):
    """
    Applies the sliding window method to the data supplied by format_data(). The return will be a list
    of pandas DataFrames:

    - each DataFrame represents a meter
    - rows represent 9-tuples data points represented by columns
    - columns: 7 days + peak_position + peak_value
    """

    feature_matrix_list = []

    # Iterating over rows
    for i, row in data.iterrows():
        # Strip the number of data points used for aggregation
        meter_data = np.array([tuple_[0] for tuple_ in row])

        # Construct the list of data frames for each user
        feature_matrix_meter = apply_sliding_window(meter_data)
        feature_matrix_list.append(feature_matrix_meter)

    return feature_matrix_list

def prev_week_DataFrame(meter_data):
    """
    Applies the 'PrevWeek' method to obtain peak positions and peak values.
    It is applied to a Data Frame containing the data for a single meter.
    It returns a copy of the initial data frame, with two extra columns: 

    - "PrevWeek Peak Value": the predicted value via PrevWeek
    - "PrevWeek Peak Position": the predicted value via PrevWeek
    """

    dropped_data = meter_data.drop(columns=["Peak Value", "Peak Position"])

    meter_data["PrevWeek Peak Value"] = dropped_data.apply(np.max, axis=1) # Axis 1: max per each row
    meter_data["PrevWeek Peak Position"] = dropped_data.apply(np.argmax, axis=1) 

    return meter_data

# --- Roberto's implementation of the sliding window & prev week

def construct_feature_matrix_array(data: pd.DataFrame):
    """
    TODO: Calculate the peak value and peak position labels of the training part of the dataset

    Applies the sliding window method to the data supplied by format_data(). The return will be a list
    of pandas DataFrames:

    - each DataFrame represents a meter
    - rows represent 9-tuples data points represented by columns
    - columns: 7 days + peak_position + peak_value
    """

    master_feature_matrix = []

    for i in range(len(data.index)):

        meter_time_series = data.iloc[i].values

        lenght_time_series = len(meter_time_series)

        meter_time_series = np.array([meter_time_series[j][0] for j in range(lenght_time_series)])


        window_data = 7
        window_peak = 7

        for j in range(lenght_time_series - window_data-window_peak-1):

            week_data = meter_time_series[j:j + window_data]
            week_peak = meter_time_series[j + window_data:j + window_data + window_peak]
            peak_value = np.max(week_peak)
            peak_position = np.argmax(week_peak) % 7
            temp = np.array([peak_value, peak_position])
            feature = np.concatenate((week_data, temp))

            if j == 0:
                feature_matrix = feature

            feature_matrix = np.vstack((feature_matrix, feature))


        master_feature_matrix.append(feature_matrix)

    return master_feature_matrix


def prev_week_array(feature_matrix):

    size = feature_matrix.shape[0]
    peak_value_array = np.zeros(size)
    peak_position_array = np.zeros(size)

    for i in range(size):

        week_values = feature_matrix[i, 0:7]
        peak_value = np.max(week_values)
        peak_position = np.argmax(week_values) % 7

        peak_value_array[i] = peak_value
        peak_position_array[i] = peak_position

    return peak_value_array, peak_position_array