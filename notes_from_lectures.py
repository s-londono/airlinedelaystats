df.groupby(['CompanySize']).mean()['JobSatisfaction'].sort_values()

# Drop only missing values
all_row = small_dataset.dropna(axis=0, how='all')

only3_drop = small_dataset.dropna(subset=['col3'], how='any')

# Drop a variable (column)
df_drop.drop('reports', axis=1)

# Drop all rows or columns wit
# h all values null (axis=0 rows, axis=1 columns)
reduced_df7 = df_drop.dropna(axis=0, how='all')

# Drop all rows or columns with a specific column has null any values (axis=0 rows, axis=1 columns)
reduced_df8 = df_drop.dropna(subset=['name'], axis=0)

# Drop rows where several columns match a condition
reduced_df8 = df_drop.dropna(subset=['name', 'year'], axis=0)

# Proportion of individuals in the dataset with salary reported
prop_sals = 1 - df.isnull()['Salary'].mean()


# Normalize the data. Simplifies modeling
scaler = preprocessing.MinMaxScaler()
arr_delays_scaled = scaler.fit_transform(df_all_delays["ArrDelay"].values.reshape(-1, 1))

plt.hist(arr_delays_scaled)
plt.show()