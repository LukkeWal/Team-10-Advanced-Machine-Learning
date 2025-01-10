"""
Prepares the data from https://datadryad.org/stash/dataset/doi:10.5061/dryad.m0cfxpp2c (22-11-2024) 
to be used for machine learning purposes. It requires the data from the before mentioned link to be 
unzipped, and the folder which contains the csv files should be placed in the same folder as this file
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from DataStatistics import date_complete_cell_percentage, date_percentage_duplicate, meter_complete_cell_percentage, meter_percentage_consecutive_duplicate

##### Data preprocessing: loading raw files, bringing to desired format in one csv, saving raw files #####

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
        try:
            meter_data["INTERVAL_TIME"] = meter_data["INTERVAL_TIME"].apply(string_to_datetime)
        except ValueError as e:
            print(f"skipping {current_meter_num} from data due to malformed datetime: {e}")
            continue
        meter_data["DATE"] = meter_data["INTERVAL_TIME"].dt.date  # Extract date

        # Aggregate by DATE: sum measurements and count occurrences
        daily_data = meter_data.groupby("DATE").agg(
            SUM_MEASUREMENTS=("INTERVAL_READ", "sum"),
            AMOUNT_OF_MEASUREMENTS=("INTERVAL_READ", "count")
        ).reset_index()
        result.append(daily_data)
    print("Summing measurements to 1 day intervals... DONE")
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

def remove_bad_dates(data: pd.DataFrame):
    # get the bad dates
    _, bad_dates_completeness = date_complete_cell_percentage(data, acceptance_range=5, remove_threshold=90)
    _, bad_dates_duplication = date_percentage_duplicate(data, remove_threshold=10)
    bad_dates = bad_dates_completeness | bad_dates_duplication

    # to give more information to the user we check how many groups 
    # of consecutive dates are being removed and we print them
    bad_dates = sorted(bad_dates)
    group_count = 1
    timeskip_dates = []
    for i in range(1, len(bad_dates)):
        if bad_dates[i] - bad_dates[i-1] > timedelta(days=1):
            group_count += 1  # New group starts if dates are not consecutive
            timeskip_dates.append(bad_dates[i])
    print(f"we are removing {len(bad_dates)} / {len(data.columns)} columns. There will be {group_count} timeskips at:")
    # for t in timeskip_dates:
    #     print(f"\t{t.date()}")
    for date in bad_dates:
        data = data.drop(date, axis=1)
    return data

def remove_bad_meters(data: pd.DataFrame):
    # the removal parameters are harsher on meters than dates because less meters has less impact compared to less dates
    _, bad_meters_completeness = meter_complete_cell_percentage(data, acceptance_range=0, remove_threshold=99)
    _, bad_meters_duplication = meter_percentage_consecutive_duplicate(data, remove_threshold=1)
    bad_meters = bad_meters_completeness | bad_meters_duplication
    print(f"We are removing {len(bad_meters)}/{len(data.columns)}, {len(bad_meters_duplication)} for duplication and {len(bad_meters_completeness)} for low competeness")
    for index in bad_meters:
        data = data.drop(index)
    return data

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
    data = remove_bad_dates(data) # remove dates(columns) that have missing values or a suspicious amount of duplicate values
    data = remove_bad_meters(data)
    #TODO: data = z_normalize(data)
    data.to_csv(save_filename, index=False) # save results becaues its a lot of data and takes long
    return data

##### Data processing: Sliding window (with skipping dates), normalization, saving and loading #####

# --- Andrei's implementation (DataFrame focused)

def skip_dates_mask(data: pd.DataFrame,
                    length_window_data = 7, length_window_peak = 7):
    """
    This function constructs an array of booleans which dictate which windows to be skipped
    (due to dates having been skipped within the window). It works on the DataFrame supplied by
    format_data()

    True = Skip window ; False = Continue with window

    The underlying principle is that all meters need to skip the same dates. That is, missing
    dates are the same for all users. So a mask for one user is a mask for all.
    """
 
    dates = data.columns  # Extract all the dates
    dates = dates.to_pydatetime()
    mask = []

    for i in range(len(dates) - length_window_peak- length_window_data + 1):
        dates_window = dates[i:(i+length_window_data+length_window_peak)]
        bad_date_status = False

        for i in range(len(dates_window)-1):
            difference = dates_window[i+1] - dates_window[i]

            if difference.days != timedelta(days=1).days: # if two dates are consecutive in the array but not temporally consecutive we have deleted one
                bad_date_status = True
                break  # Stop checking if we hit a faulty date already

        mask.append(bad_date_status)
    
    return np.array(mask)

def apply_sliding_window(meter_data, dates_mask, normalize=True):
    """
    Given the data **for a single meter**, this function constructs another DataFrame whose colummns are
    
    - the 7 days: week on which we predict
    - (peak value, peak position): belonging to the next week

    TODO: This function further checks if dates have been skipped and performs z-normalization 
    if normalization = True.
    """

    length_window_data = 7    # Lenght of window which makes the covariates
    length_window_peak = 7    # Lenght of window from which we extract peaks
    len_series = len(meter_data)
    
    # Normalize
    if normalize == True:
        meter_data = (meter_data - meter_data.mean() * np.ones(len_series)) / meter_data.std()

    # Data frame of the desired format
    colnames = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", 
    "Peak Value", "Peak Position"]
    feature_matrix_df = pd.DataFrame(columns = colnames)

    for i in range(len_series - length_window_peak- length_window_data + 1):
        # Check whether to skip the window or not
        if dates_mask[i] == True:
            continue

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

def construct_feature_matrix_DataFrame(data: pd.DataFrame, normalize=True):
    """
    Applies the sliding window method to the data supplied by format_data(). The return will be a list
    of pandas DataFrames:

    - each DataFrame represents a meter
    - rows represent 9-tuples data points represented by columns
    - columns: 7 days + peak_position + peak_value

    If normalize = True, the data within each meter is z-normalized.
    """

    feature_matrix_list = []
    dates_mask = skip_dates_mask(data)  # Mask with the windows to be skipped by the sliding window method

    # Iterating over rows
    for i, row in data.iterrows():
        # Strip the number of data points used for aggregation
        meter_data = np.array([tuple_[0] for tuple_ in row])
        
        # Construct the list of data frames for each user
        feature_matrix_meter = apply_sliding_window(meter_data, dates_mask, normalize=normalize)
        feature_matrix_list.append(feature_matrix_meter)

    return feature_matrix_list

def load_and_save_processed_data(precomp_data_dirname="time_series_matrix.csv",
                                 save_data=False) -> pd.DataFrame:
    """
    Loads the precomputed data (as returned by format_data()) from one of the 
    csv files. It returns the processed data (after z-normalization and sliding window)
    (same format as construct_feature_matrix_DataFrame()),
    and it has the option to save for each user its respective Data Frame.

    NOTE: To read the precomputed data, it suffices to use the read_data("ProcessedData")

    TODO: Create the saving data function
    """

    # Load the full data set
    data = load_precomputed_data(precomp_data_dirname)   
    # Compute list of DataFrames: each Df for a user. 
    # Data frame in desired format ready to plug 
    data_processed = construct_feature_matrix_DataFrame(data, normalize=True)

    if save_data == True: # save results becaues its a lot of data and takes long
        for i in range(len(data_processed)):
            processed_meter_df = data_processed[i]
            file_path = os.path.join('ProcessedData', f'{i}.csv')
            processed_meter_df.to_csv(file_path, index=False)

    return data_processed

# --- Roberto's implementation (array focused)

def check_skipped_dates(date_array):

    """
    The function returns True if within the given array of dates, a column has been deleted.
    That is, the function returns True if there exist two consecutive dates (in the array)
    that are not temporally consecutive.

    """

    bad_date_status = False

    for i in range(len(date_array)-1):

        day1 = datetime.strptime(date_array[i], '%Y-%m-%d').date()
        day2 = datetime.strptime(date_array[i + 1], '%Y-%m-%d').date()

        difference = day2 - day1

        if difference.days != timedelta(days=1).days: # if two dates are consecutive in the array but not temporally consecutive we have deleted one
            bad_date_status = True
            break  # Stop checking if we hit a faulty date already

    return bad_date_status


def construct_feature_matrix_array(data: pd.DataFrame):
    """

    Applies the sliding window method to the data supplied by format_data(). The return will be a list
    of matrices:

    - each matrix represents a meter
    - rows represent 9-tuples data points represented by columns
    - columns: 7 days + peak_position + peak_value
    """

    master_feature_matrix = []

    dates = data.columns

    for i in range(len(data.index)):


        meter_time_series = data.iloc[i].values

        lenght_time_series = len(meter_time_series)

        meter_time_series = np.array([  eval(meter_time_series[j])[0] for j in range(lenght_time_series)])


        window_data = 7
        window_peak = 7

        meter_time_series = (meter_time_series - meter_time_series.mean())/(meter_time_series.std()) # normalize the time series

        for j in range(lenght_time_series - window_data-window_peak-1): # +1 ?

            week_data = meter_time_series[j:j + window_data]

            if check_skipped_dates(dates[j:j+window_data]) == True:
                continue # we skip if there is a bad date

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

##### Baselines: PrevWeek #####

# Andrei
def prev_week_DataFrame(meter_data):
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

    meter_data["PrevWeek Peak Value"] = dropped_data.apply(np.max, axis=1) # Axis 1: max per each row
    meter_data["PrevWeek Peak Position"] = dropped_data.apply(np.argmax, axis=1) 

    #return meter_data

    peak_value = meter_data["PrevWeek Peak Value"].values
    peak_position = meter_data["PrevWeek Peak Position"].values

    return peak_value, peak_position

# Roberto
def prev_week_array(feature_matrix):

    """
    Applies the 'PrevWeek' method to obtain peak positions and peak values.
    It is applied to a Data Frame containing the data for a single meter.
    It returns a copy of the initial data frame, with two extra columns:

    - "PrevWeek Peak Value": the predicted value via PrevWeek
    - "PrevWeek Peak Position": the predicted value via PrevWeek
    """

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