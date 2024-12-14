import pandas as pd
import matplotlib.pyplot as plt

def generate_and_visualize_stats(data: pd.DataFrame):
    stat1, _ = meter_complete_cell_percentage(data)
    stat2, _ = date_complete_cell_percentage(data, acceptance_range=5, remove_threshold=90)
    stat3, _ = meter_percentage_consecutive_duplicate(data)
    stat4, _ = date_percentage_duplicate(data, remove_threshold=10)
    visualize_meter_cell_percentage(stat1)
    visualize_date_cell_percentage(stat2)
    visualize_meter_duplicates(stat3)
    visualize_date_duplicates(stat4)

def meter_complete_cell_percentage(data: pd.DataFrame, acceptance_range = 0, remove_threshold=99) -> list:
    """
    makes a list with the percentage of complete cells of every meter
    by default a cell is considered complete if it has exactly 96 measurements
    with acceptance range this can be changed into 96 +- acceptance_range
    """
    result = []
    incomplete_meters = set()
    for index, meter in data.iterrows():
        amount_of_complete_cells = 0
        for cell in meter[1:]:  # Inner loop: Iterate over cells (skip index)
            if cell[1] >= 96 - acceptance_range and cell[1] <= 96 + acceptance_range:
                amount_of_complete_cells += 1
        result.append(amount_of_complete_cells / len(meter[1:]) * 100)
        if result[-1] < remove_threshold:
            incomplete_meters.add(index)
    return result, incomplete_meters

def date_complete_cell_percentage(data: pd.DataFrame, acceptance_range = 0, remove_threshold = 99) -> list:
    """
    makes a list with the percentage of complete cells of every date
    by default a cell is considered complete if it has exactly 96 measurements
    with acceptance range this can be changed into 96 +- acceptance_range
    """
    result = []
    incomplete_dates = set()
    for date in data:
        amount_of_complete_cells = 0
        for cell in data[date]:
            if cell[1] >= 96 - acceptance_range and cell[1] <= 96 + acceptance_range:
                amount_of_complete_cells += 1
        if amount_of_complete_cells / len(data[date]) * 100 < remove_threshold:
            incomplete_dates.add(date)
        result.append(amount_of_complete_cells / len(data[date]) * 100)
    return result, incomplete_dates

def meter_percentage_consecutive_duplicate(data: pd.DataFrame, remove_threshold=1) -> list:
    result = []
    bad_meters = set()
    for index, meter in data.iterrows():
        amount_of_duplicate_cells = 0
        previous_cell = -1
        for cell in meter[1:]:  # Inner loop: Iterate over cells (skip index)
            if cell[0] == previous_cell:
                amount_of_duplicate_cells += 1
            previous_cell = cell[0]
        result.append(amount_of_duplicate_cells / len(meter[1:]) * 100)
        if result[-1] > remove_threshold:
            bad_meters.add(index)
    return result, bad_meters

def date_percentage_duplicate(data: pd.DataFrame, remove_threshold = 1) -> list:
    result = []
    dates = set()
    for date in data:
        amount_of_duplicates = data[date].duplicated(keep=False).sum()
        if amount_of_duplicates / len(data[date]) * 100 > remove_threshold:
            dates.add(date)
        result.append(amount_of_duplicates / len(data[date]) * 100)
    return result, dates

def visualize_meter_cell_percentage(percentages: list):
    # Create the boxplot
    plt.boxplot(percentages)

    # Customize the plot
    plt.xlabel("meter coverage")
    plt.ylabel("percentage of complete cells")
    plt.grid(True)  # Optional: Add a grid for better readability
    # Show the plot
    plt.show()

def visualize_date_cell_percentage(percentages: list):
    # Create the boxplot
    plt.boxplot(percentages)

    # Customize the plot
    plt.xlabel("date coverage")
    plt.ylabel("percentage of complete cells")
    plt.grid(True)  # Optional: Add a grid for better readability
    # Show the plot
    plt.show()

def visualize_meter_duplicates(percentages: list):
    # Create the boxplot
    plt.boxplot(percentages)

    # Customize the plot
    plt.xlabel("meter duplicates")
    plt.ylabel("percentage of consecutive duplicate cells")
    plt.grid(True)  # Optional: Add a grid for better readability
    # Show the plot
    plt.show()
    
def visualize_date_duplicates(percentages: list):
    # Create the boxplot
    plt.boxplot(percentages)

    # Customize the plot
    plt.xlabel("date duplicates")
    plt.ylabel("percentage of duplicate cells")
    plt.grid(True)  # Optional: Add a grid for better readability
    # Show the plot
    plt.show()