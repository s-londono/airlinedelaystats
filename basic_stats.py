import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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

# TOTAL DELAYED FLIGHTS AND CANCELLED FLIGHTS

df_all_delayed_flights_20 = df_success_flights[df_success_flights["ArrDelay"] > 20.0]
df_all_delayed_flights_1h = df_success_flights[df_success_flights["ArrDelay"] > 60.0]
df_cancelled_flights = df_failed_flights[df_failed_flights["Cancelled"] == 1.0]
df_diverted_flights = df_failed_flights[df_failed_flights["Diverted"] == 1.0]

total_flights = df_success_flights.shape[0]
total_delayed_20 = df_all_delayed_flights_20.shape[0]
total_delayed_1h = df_all_delayed_flights_1h.shape[0]
total_cancelled = df_cancelled_flights.shape[0]
total_diverted = df_diverted_flights.shape[0]

perc_delayed = (total_delayed_1h / total_flights) * 100
perc_delayed20 = (total_delayed_20 / total_flights) * 100
perc_cancelled = (total_cancelled / total_flights) * 100
perc_diverted = (total_diverted / total_flights) * 100

