import pandas as pd
import os
from datetime import datetime, timedelta
"""
Prepares the data from https://datadryad.org/stash/dataset/doi:10.5061/dryad.m0cfxpp2c (22-11-2024) 
to be used for machine learning purposes. It requires the data from the before mentioned link to be 
unzipped, and the LADPU folder which contains the csv files should be placed in the same folder as this file
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

def change_interval_to_one_day(data: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Given the original data, transform the intervals from 15 minutes to 1 day intervals.
    Saves the sum of all intervals in a day, the amount of intervals in that day and the date of the day
    """
    result = []
    total_meter_num = len(data)
    current_meter_num = 0
    for meter_data in data:
        current_meter_num += 1
        print(f"Summing measurements to 1 day intervals... {current_meter_num}/{total_meter_num}", end="\r")
        if(len(meter_data) <= 1): # some meters do not have any data, we skip these meters
            continue
        time_series_in_days = []
        # we start with the first measurement of tht sis meter
        current_date = string_to_datetime(meter_data.iloc[0]["INTERVAL_TIME"]).date()
        single_day_measurements = [meter_data.iloc[0]["INTERVAL_READ"]]
        # since we use the first measurement as a starting point we start checking from the second measurement onwards
        for index in range(1, len(meter_data)):
            # as long as the date does not change we keep appending the readings to total day measurements
            if current_date == string_to_datetime(meter_data.iloc[index]["INTERVAL_TIME"]).date():
                single_day_measurements.append(meter_data.iloc[index]["INTERVAL_READ"])
            # When the date changes save the sum of the previous day measurements and start with the next day
            else:
                time_series_in_days.append([sum(single_day_measurements), current_date, len(single_day_measurements)])
                # Add None values for any days that may have been skipped
                amount_of_jumped_days = string_to_datetime(meter_data.iloc[index]["INTERVAL_TIME"]).date() - current_date
                amount_of_jumped_days = amount_of_jumped_days.days - 1
                for i in range(amount_of_jumped_days):
                    time_series_in_days.append([None, current_date + timedelta(days=i), 0])
                # Update the single day measurements and current_date to start working on the next day
                single_day_measurements = [meter_data.iloc[index]["INTERVAL_READ"]]
                current_date = string_to_datetime(meter_data.iloc[index]["INTERVAL_TIME"]).date()
        # After completing a meter we save the last day and add this meters dataframe to results
        time_series_in_days.append([sum(single_day_measurements), current_date, len(single_day_measurements)])
        result.append(pd.DataFrame(time_series_in_days, columns=["SUM_MEASUREMENTS", "DATE", "AMOUNT_OF_MEASUREMENTS"]))
    print(f"Summing measurements to 1 day intervals... DONE")
    return result
        
def format_data(data: list[pd.DataFrame], set_missing_as_NA=True) -> pd.DataFrame:
    """
    Turn the list of dataframes into a single dataframe where every row represents a household
    and every column a day, a cell contains a households summed measurements on a specific day
    """
    # Find the earliest and latest date in our data:
    earliest = datetime.max.date()
    latest = datetime.min.date()
    total_meter_num = len(data)
    current_meter_num = 0
    for meter_data in data:
        current_meter_num += 1
        print(f"Calculating start date and end date of measurements... {current_meter_num}/{total_meter_num}", end="\r")
        first_date = meter_data.iloc[0]["DATE"]
        last_date = meter_data.iloc[-1]["DATE"]
        if last_date > latest:
            latest = last_date
        if first_date < earliest:
            earliest = first_date
    print(f"Calculating start date and end date of measurements... DONE")
    print(f"earliest date: {earliest}")
    print(f"latest date: {latest}")

    # Generate a list from earliest to latest to serve as our columns
    print(f"Generating column labels... ", end="\r")
    dates = []
    difference_in_days = latest - earliest
    difference_in_days = difference_in_days.days
    for i in range(difference_in_days+1):
        dates.append(earliest + timedelta(days=i))
    print(f"Generating column labels... DONE")

    # Combine the measurements of the meters into a single matrix
    result = []
    meter_progresses = len(data) * [0]
    total_days = len(dates)
    current_day = 0
    # We go column by column
    for date in dates:
        current_day += 1
        print(f"Formatting matrix... {round(current_day/total_days*100)}%", end="\r")
        column = []
        # For every column we get its corresponding value (or None) of the meters on that date
        for meter_index in range(len(data)):
            meter_progress = meter_progresses[meter_index]
            meter_data = data[meter_index]
            if meter_progress == 0: # The start date of this meter has not been reached yet, check if we have reached it now
                if meter_data.iloc[0]["DATE"] == date: # we have reached the start date of this meter
                    if set_missing_as_NA == True and meter_data.iloc[0]["AMOUNT_OF_MEASUREMENTS"] < 96:
                        column.append(None)
                    else:
                        column.append(meter_data.iloc[0]["SUM_MEASUREMENTS"])
                    meter_progresses[meter_index] += 1
                else: # We have not reached the start date of this meter, add None
                    column.append(None)
            elif meter_progress == -1: # The end date of this meter has been passed
                column.append(None)
            else: # date is in the range of this meter, add datapoint to matrix
                if(meter_progress < len(meter_data)):
                    if set_missing_as_NA == True and meter_data.iloc[0]["AMOUNT_OF_MEASUREMENTS"] < 96:
                        column.append(None)
                    else:
                        column.append(meter_data.iloc[meter_progress]["SUM_MEASUREMENTS"])
                    meter_progresses[meter_index] += 1
                else:
                    column.append(None)
                    meter_progresses[meter_index] = -1
        result.append(column)
    print(f"Formatting matrix... DONE")
    result = pd.DataFrame(list(zip(*result)), columns=dates)
    return result



            
            

    return

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

def z_normalize():
    """
    TODO: z normalize the given time series
    """
    return

def split_train_and_test():
    """
    TODO: Split the data into a training set and a test set. This test set should not be used
    for anything but the final benchmark of our project
    """
    return

def calculate_labels():
    """
    TODO: Calculate the peak value and peak position labels of the training part of the dataset
    """
    return


