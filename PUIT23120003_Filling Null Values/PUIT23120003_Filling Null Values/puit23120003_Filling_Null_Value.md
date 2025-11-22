Import Required Libraries

``` python
import pandas as pd # for data manuipulation and analysis
import numpy as np # for data manuipulation and analysis
import sklearn as sk # for Machine learning metrics
import matplotlib.pyplot as plt # for Plotting
```

Load Your Dataset

``` python
df = pd.read_csv('/content/sensor_log.csv')
```

Viewing dataset

``` python
df.head()
```

Basic information about the dataset

``` python
info = {
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
    "missing_counts": df.isna().sum().to_dict(),
    "total_missing": int(df.isna().sum().sum())
}
print("Basic info:", info)
```

Find duplicates (all columns identical)

``` python
duplicates_mask = df.duplicated(keep=False)
duplicates = df[duplicates_mask].copy()
duplicates_count = duplicates.shape[0]
print(f"Found {duplicates_count} duplicate rows (keep=False).")
```

Numeric columns for imputation evaluation

``` python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", numeric_cols)
```

Handle Missing Values - Interpolation

``` python
df_interpolated = df.interpolate()
df_interpolated
```

Rechecking for missing values

``` python
df_interpolated.isna().sum()
```

Handle Missing Values - Forward Fill

``` python
df_ffill = df.fillna(method='ffill')
df_ffill
```

Rechecking for missing values

``` python
df_ffill.isna().sum()
```

Handle Missing Values - Backward Fill

``` python
df_bfill = df.fillna(method='bfill')
df_bfill
```

Rechecking for missing values

``` python
df_bfill.isna().sum()
```

Finding rows that originally had missing values

``` python
missing_positions = list(zip(*np.where(df.isna())))
print("\nRows that had missing values:", missing_positions)
```

Comparing the 3 methods

``` python
results = []
for row, col in missing_positions:
    colname = df.columns[col]  # get column name

    true_val = df_interpolated.loc[row, colname]   # reference value from interpolation
    ffill_val = df_ffill.loc[row, colname]        # forward fill value
    bfill_val = df_bfill.loc[row, colname]        # backward fill value

    results.append({
        'row': row,
        'column': colname,
        'interpolated_value': true_val,
        'ffill_value': ffill_val,
        'bfill_value': bfill_val,
        'ffill_error': abs(ffill_val - true_val),
        'bfill_error': abs(bfill_val - true_val)
    })

comp_df = pd.DataFrame(results)
print("\nComparison Table:")
comp_df
```

Showing total and average errors

``` python
print("\n==================== ERROR SUMMARY ====================")
print("Total Forward Fill Error:", comp_df['ffill_error'].sum())
print("Total Backward Fill Error:", comp_df['bfill_error'].sum())
print("Average Forward Fill Error:", comp_df['ffill_error'].mean())
print("Average Backward Fill Error:", comp_df['bfill_error'].mean())

print("\nBEST METHOD = Interpolation (0 error, smooth & realistic)")
```

Saving Clean version

``` python
df_ffill.to_csv("sensor_log_ffill.csv", index=False)
df_bfill.to_csv("sensor_log_bfill.csv", index=False)
df_interpolated.to_csv("sensor_log_interpolated.csv", index=False)

print("\nCleaned files saved:")
print(" - sensor_log_ffill.csv")
print(" - sensor_log_bfill.csv")
print(" - sensor_log_interpolated.csv")
```
