import pandas as pd
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

def split_train_and_test( dataframe ):

    nRows, nCols = dataframe.shape
    trainSize = int( nRows  * 0.8)
    trainData =  dataframe.iloc[:trainSize,:]
    testData = dataframe.iloc[trainSize:,:]


    return trainData, testData

def calculate_labels():
    """
    TODO: Calculate the peak value and peak position labels of the training part of the dataset
    """
    return

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
