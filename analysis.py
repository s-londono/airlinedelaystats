import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import Utilities as util

# (1936758, 30)
df_flight_delays = pd.read_csv("datasets/DelayedFlights.csv")
print(f"Airline Delays: {df_flight_delays.shape}")

# Drop the first column, which just provides enumerates the rows. The DataFrame index does so already
df_flight_delays.drop(df_flight_delays.columns[:2], axis=1, inplace=True)

# Split datasets
df_success_flights = df_flight_delays[(df_flight_delays["Cancelled"] == 0.0) &
                                      (df_flight_delays["Diverted"] == 0.0)].copy()

df_failed_flights = df_flight_delays[(df_flight_delays["Cancelled"] == 1.0) |
                                     (df_flight_delays["Diverted"] == 1.0)].copy()

df_success_flights.dropna(subset=["TailNum"], inplace=True)

# Drop rows having missing values for the TailNum variable
df_success_flights.dropna(subset=["TailNum"], inplace=True)

# In this part, we care only about delay-related variables
df_all_delay_data = df_success_flights[["ArrDelay", "CarrierDelay", "WeatherDelay", "NASDelay",
                                        "SecurityDelay", "LateAircraftDelay"]]

# Get flights that have complete information about delays
df_flights_full_dly_data = df_all_delay_data[(df_success_flights["ArrDelay"].isna() == False) &
                                             (df_success_flights["CarrierDelay"].isna() == False) &
                                             (df_success_flights["WeatherDelay"].isna() == False) &
                                             (df_success_flights["NASDelay"].isna() == False) &
                                             (df_success_flights["SecurityDelay"].isna() == False) &
                                             (df_success_flights["LateAircraftDelay"].isna() == False)]

print(f"Total flights not missing delay information: {df_flights_full_dly_data.shape[0]}")

# Accumulate delays of each type and arrival delays for all flights with full delay information
df_delay_totals = df_flights_full_dly_data.sum()

# For the totals of each type of delay, compute the ratio to the total arrival delay
df_dly_weights = df_delay_totals / df_delay_totals["ArrDelay"]

# Impute missing values for each delay category variable, by splitting up the arrival delay as to the weights
for var in ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]:
    df_success_flights[var].fillna(df_success_flights["ArrDelay"] * df_dly_weights[var], inplace=True)

# Check that there are no missing values left
util.build_nan_data(df_success_flights)

df_success_flights_frag = df_success_flights.iloc[0:50, :]


# DELAY CAUSES

gr_sccs_flights_x_carrier = df_success_flights.groupby(by="UniqueCarrier")

df_total_delays_x_carrier = pd.DataFrame({"ArrDelay": gr_sccs_flights_x_carrier["ArrDelay"].sum(),
                                          "CarrierDelay": gr_sccs_flights_x_carrier["CarrierDelay"].sum(),
                                          "WeatherDelay": gr_sccs_flights_x_carrier["WeatherDelay"].sum(),
                                          "NASDelay": gr_sccs_flights_x_carrier["NASDelay"].sum(),
                                          "SecurityDelay": gr_sccs_flights_x_carrier["SecurityDelay"].sum(),
                                          "LateAircraftDelay": gr_sccs_flights_x_carrier["LateAircraftDelay"].sum()})

df_mean_delays_x_carrier = pd.DataFrame({"ArrDelay": gr_sccs_flights_x_carrier["ArrDelay"].mean(),
                                         "CarrierDelay": gr_sccs_flights_x_carrier["CarrierDelay"].mean(),
                                         "WeatherDelay": gr_sccs_flights_x_carrier["WeatherDelay"].mean(),
                                         "NASDelay": gr_sccs_flights_x_carrier["NASDelay"].mean(),
                                         "SecurityDelay": gr_sccs_flights_x_carrier["SecurityDelay"].mean(),
                                         "LateAircraftDelay": gr_sccs_flights_x_carrier["LateAircraftDelay"].mean()})

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(15, 7))

sns.set_color_codes("pastel")
sns.barplot(x="CarrierDelay", y=0, data=df_mean_delays_x_carrier, label="Carrier", color="b")

# Heatmap of correlation

# Get a correlation metric and render as a Heat Map
corr_mtx = df_success_flights[["DepDelay", "ArrDelay", "CarrierDelay", "WeatherDelay",
                              "NASDelay", "SecurityDelay", "LateAircraftDelay"]].corr()

sns.heatmap(corr_mtx, annot=True, fmt=".2f")

# https://seaborn.pydata.org/examples/index.html
# https://seaborn.pydata.org/examples/horizontal_barplot.html
# https://seaborn.pydata.org/examples/horizontal_boxplot.html
# https://matplotlib.org/2.0.0/api/pyplot_api.html#matplotlib.pyplot.axis
