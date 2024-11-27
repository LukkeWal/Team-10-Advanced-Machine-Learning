# Simplyfing and optimizing reading the raw data:
 ## using pandas to_datetime instead of my custom function
    using my own datatime translator: 83 sec
    using pandas datetime translator: 279 sec
## using padas group_by to sum intervals to 1 day
    using my own sum method: 83 sec
    using group_by: 8 sec
    will use group_by from now on