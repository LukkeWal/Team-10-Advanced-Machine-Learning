# Shape of final time series data
- The final data format is a matrix where every row represents a household/energy meter and every column is a date. The value in a single cell is the sum of all datapoints on that day, or NA if there were no datapoints. 

# Future work
- I have not done the "remove first year of missing values" step since the data is a bit complexer? there seems to be 7 years worth of data so have to look at it in more detail later
- We still have to work on the values in he final matrix cells, since currently some cells have the sum of 96 datapoints (1 every 15 minutes) and some less

# Some perculiarities in our dataset
- when looking at the date+time of measurements some measurements have their full date and time, while others only have the date
- some dates use the full 4 digit year, while other use a 2 digit year