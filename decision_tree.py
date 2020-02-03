import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
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

# TODO: REMOVE DepDelay, which is colinear with ArrDelay

# Take a random sample. Dataset is too big
df_success_flights_sample = df_success_flights.sample(n=100000, axis=0)

df_nas_sample = df_success_flights_sample.isna().sum()

# Remove variables that are redundant or that are unknown prior to the flight
# df_success_flights_x_linmod = df_success_flights_sample[["Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime",
#                                                          "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime",
#                                                          "ArrDelay", "Origin", "Dest", "Distance"]]

df_success_flights_x_linmod = df_success_flights_sample[["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier",
                                                         "ArrDelay", "Origin", "Dest", "Distance"]]

# dict_months = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
#                11: "Nov", 12: "Dec"}
#
# df_success_flights_x_linmod.loc[:, "Month"] = df_success_flights_x_linmod["Month"].transform(lambda v: dict_months[v])

df_success_flights_x_linmod = \
    df_success_flights_x_linmod.astype({"Month": "str", "DayofMonth": "str", "DayOfWeek": "str"})

# Encode categorical variables:
cat_vars = df_success_flights_x_linmod.select_dtypes(include=["object"]).columns.values
df_processed = pd.get_dummies(df_success_flights_x_linmod, prefix=cat_vars, columns=cat_vars, drop_first=True)

# Target variable is ArrDelay (arrival delay)
X = df_processed[np.setdiff1d(df_processed.columns.values, ["ArrDelay"])]
y = df_processed["ArrDelay"]

# Split the data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Create a Decision Tree Regression model and fit to the dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Predict on the test data
y_test_preds = regressor.predict(X_test)

# Score the model on the test data
# R-squared value to measure how well the predicted values compare to the actual test values
r2_test = r2_score(y_test, y_test_preds)

print(r2_test)
