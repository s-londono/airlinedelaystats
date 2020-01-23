import pandas as pd

# (1936758, 30)
df_airline_delay = pd.read_csv("datasets/DelayedFlights.csv")
print(f"Airline Delays: {df_airline_delay.shape}")

print(df_airline_delay.columns)

print(df_airline_delay['Year'].value_counts())

# Total missing values in
df_nan_counts = df_airline_delay.isna().sum()

# Rows with all values missing
nan_counts_row = df_airline_delay.isna().sum(axis=1)

# Total rows with more than 35% of the values missing
sum((nan_counts_row / df_airline_delay.shape[1]) > 0.35)

# DataFrame of rows with more than 35% of the values missing
df_rows_with_many_nans = df_airline_delay.loc[(nan_counts_row / df_airline_delay.shape[1]) > 0.35, :]

# Identify categorical variables
ix_catg_cols = df_airline_delay.select_dtypes(include=["object"]).columns
ix_numr_cols = df_airline_delay.select_dtypes(include=["number"]).columns
ix_bool_cols = df_airline_delay.select_dtypes(include=["bool"]).columns

# Note however, that month and day of week are actually categorical and cancelled is boolean. What else?

# Drop columns having the same value in every row (too slow)
val_counts = df_airline_delay.apply(lambda c: c.value_counts())

