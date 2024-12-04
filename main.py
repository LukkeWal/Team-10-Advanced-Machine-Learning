import time

from PrepareData import load_and_save_raw_data, load_precomputed_data
from DataStatistics import generate_and_visualize_stats, plot_time_series


def main():
    start_time = time.time()
    #data = load_and_save_raw_data("LADPU", "time_series_matrix.csv")
    data = load_precomputed_data("time_series_matrix_TESTING.csv")
    plot_time_series(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"loaded data in {elapsed_time:.4f} seconds")
    #generate_and_visualize_stats(data)
    return

if __name__ == "__main__":
    main()