import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

# (1936758, 30)
df_flight_delays = pd.read_csv("datasets/DelayedFlights.csv")
print(f"Airline Delays: {df_flight_delays.shape}")

# Extract subset
# ds_subset = df_airline_delay.iloc[:500]
# ds_subset.to_csv("datasets/DelayedFlightsSub.csv")

print(df_flight_delays.columns)

print(df_flight_delays['Year'].value_counts())

# Info about variables
df_flight_delays.info()

# Compute stats about each column
df_flight_delays.describe()

# Drop the first column, which just provides enumerates the rows. The DataFrame index does so already
df_flight_delays.drop(df_flight_delays.columns[:2], axis=1, inplace=True)

# Histogram of some variables. Warn: heavy outliers
df_flight_delays[["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]].hist(bins=25)

# Heatmap of correlation
sns.heatmap(df_flight_delays.corr(), annot=True, fmt='.2f')

# Count of rows and columns
num_rows = df_flight_delays.shape[0]
num_cols = df_flight_delays.shape[1]

# DataFrame marking missing values
df_nans = df_flight_delays.isna()

# Total missing values in each column
df_nan_counts = df_nans.sum()
df_nan_percents = (df_nan_counts / df_flight_delays.shape[0]) * 100
df_nan_data = pd.DataFrame({"CountMissing": df_nan_counts, "PercentMissing": df_nan_counts}, index=df_nan_counts.index)

# Analyze missing values in ArrTime
df_missing_arrtime = df_flight_delays[df_flight_delays["ArrTime"].isna()]
sum((df_missing_arrtime["Cancelled"] == 1.0) | (df_missing_arrtime["Diverted"] == 1.0))

df_missing_any = df_flight_delays[(df_flight_delays["ActualElapsedTime"].isna()) |
                                  (df_flight_delays["AirTime"].isna()) |
                                  (df_flight_delays["ArrDelay"].isna()) | df_flight_delays["AirTime"].isna()]

df_missing_all = df_flight_delays[(df_flight_delays["ActualElapsedTime"].isna()) &
                                  (df_flight_delays["AirTime"].isna()) &
                                  (df_flight_delays["ArrDelay"].isna()) & df_flight_delays["AirTime"].isna()]

df_missing_any.head()

df_canceld_or_divertd = df_flight_delays[(df_flight_delays["Cancelled"] == 1.0) | (df_flight_delays["Diverted"] == 1.0)]

df_non_delayed = df_flight_delays[(df_flight_delays["ArrDelay"] <= 0) & (df_flight_delays["DepDelay"] <= 0)]

df_all_dly_details_na = df_flight_delays[df_flight_delays["CarrierDelay"].isna() &
                                         df_flight_delays["WeatherDelay"].isna() & df_flight_delays["NASDelay"].isna() &
                                         df_flight_delays["SecurityDelay"].isna() &
                                         df_flight_delays["LateAircraftDelay"].isna()]

df_any_dly_details_na = df_flight_delays[df_flight_delays["CarrierDelay"].isna() |
                                         df_flight_delays["WeatherDelay"].isna() | df_flight_delays["NASDelay"].isna() |
                                         df_flight_delays["SecurityDelay"].isna() |
                                         df_flight_delays["LateAircraftDelay"].isna()]

# Do missing delay columns correspond to cancelled flights?
total_cancelled = sum(df_flight_delays["Cancelled"] > 0.0)
print("Delay missing: {}. Cancelled flights: {}".format(df_nan_counts["CarrierDelay"], total_cancelled))

df_missing_carrier_delay = df_flight_delays[df_flight_delays["CarrierDelay"].isna()]

# Columns that have no missing values
ix_cols_no_nans = df_flight_delays.columns[df_nans.sum() == 0]

# Columns with all values missing (should be dropped using dropna). None
ix_cols_all_nans = df_flight_delays.columns[df_nans.sum() == num_rows]

# Rows with all values missing
nan_counts_row = df_nans.sum(axis=1)

# Total rows with more than 35% of the values missing
sum((nan_counts_row / num_cols) > 0.35)

# DataFrame of rows with more than 35% of the values missing
df_rows_with_many_nans = df_flight_delays[(nan_counts_row / num_cols) > 0.35]

# Identify categorical variables
ix_catg_cols = df_flight_delays.select_dtypes(include=["object"]).columns
ix_numr_cols = df_flight_delays.select_dtypes(include=["number"]).columns
ix_bool_cols = df_flight_delays.select_dtypes(include=["bool"]).columns

# Note however, that month and day of week are actually categorical and cancelled is boolean. What else?

# Drop columns having the same value in every row (too slow)
val_counts = df_flight_delays.apply(lambda c: c.value_counts())

# How do ArrDelay and DepDelay relate to the rest of delays?
df_delay_data = df_flight_delays.iloc[:100][["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay",
                                             "NASDelay", "SecurityDelay", "LateAircraftDelay"]]

df_delay_data["NetDelay"] = df_flight_delays[["CarrierDelay", "WeatherDelay", "NASDelay",
                                              "SecurityDelay", "LateAircraftDelay"]].sum(axis=1)

plt.plot(df_delay_data.index.values, df_delay_data["ArrDelay"], "b")
plt.plot(df_delay_data.index.values, df_delay_data["DepDelay"], "r")
plt.plot(df_delay_data.index.values, df_delay_data["NetDelay"], "g")
plt.show()

# Note that not using copy results in a strange warning. Slicing somehow causes the result to be a view and not a copy
# This difference is not always right, as ArrTime/CRSArrTime are given as hhmm. But it's enough to conclude that
# ArrDelay is the difference between ArrTime and CRSArrTime
df_cmp_arr_delay = df_flight_delays[["ArrTime", "CRSArrTime", "ArrDelay"]].copy()
df_cmp_arr_delay["DiffArrTime"] = (df_cmp_arr_delay["ArrTime"] - df_cmp_arr_delay["CRSArrTime"])


# DESCRIPTIVE STATISTICS

df_all_delays = df_flight_delays[["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay",
                                  "SecurityDelay", "LateAircraftDelay"]]

# From the stats note that the max values are pretty extreme for all types of delay (1 order of magnitude above 99prc)
# There should not be too many of those extreme outliers. Analyze them further and could be discarded?
delay_stats = df_all_delays.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99])

df_all_delays_nooutl = df_all_delays[(df_all_delays["ArrDelay"] < 257) &
                                     (df_all_delays["DepDelay"] < 257)]


def boxplot_column(col_name, title=None):
    flierprops = dict(marker=',', markerfacecolor='green', markersize=2, linestyle='none')

    plt.boxplot(df_all_delays_nooutl[col_name].dropna(),
                labels=[col_name], showmeans=True, meanline=True, flierprops=flierprops)

    plt.title(title)
    plt.show()


# All of the delays have extreme outliers (high). Show variance?
boxplot_column("ArrDelay", "Arrival Delay (minutes)")

plt.hist(df_all_delays_nooutl["ArrDelay"], bins=100)
plt.title("Arrival Delay (minutes)")
plt.show()

boxplot_column("DepDelay", "Departure Delay (minutes)")

plt.hist(df_all_delays_nooutl["DepDelay"])
plt.title("Departure Delay (minutes)")
plt.show()

df_tnsfmd_arr_logs = df_all_delays["ArrDelay"].transform(lambda v: np.log(v) if v > 0 else 0)
plt.hist(df_tnsfmd_arr_logs, bins=30)
plt.title("Arrival Delay (minutes)")
plt.show()

plt.boxplot(df_tnsfmd_arr_logs)
plt.show()

df_tnsfmd_dep_logs = df_all_delays["DepDelay"].transform(lambda v: np.log(v) if v > 0 else 0)
plt.hist(df_tnsfmd_dep_logs, bins=30)
plt.title("Departure Delay (minutes)")
plt.show()

plt.boxplot(df_tnsfmd_dep_logs)
plt.show()

# Put all heavy outliers under the same category
plt.hist(df_all_delays["ArrDelay"].transform(lambda v: v if v <= 257 else 260), bins=50)
plt.title("Arrival Delay (minutes)")
plt.show()

# Put all heavy outliers under the same category
plt.hist(df_all_delays["DepDelay"].transform(lambda v: v if v <= 249 else 260), bins=50)
plt.title("Departure Delay (minutes)")
plt.show()

