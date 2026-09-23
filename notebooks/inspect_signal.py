import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("data/raw/PowerLoad_Dataset.csv", parse_dates=["Timestamp"])
df = df.set_index("Timestamp")
target = "Power_Load_kW"

print("=== CORRELATION OF EACH COLUMN WITH Power_Load_kW ===")
print(df.corr()[target].drop(target).sort_values())

print("\n=== AVERAGE LOAD BY HOUR OF DAY ===")
print(df.groupby(df.index.hour)[target].agg(["mean", "std", "count"]).round(1))

print("\n=== AVERAGE LOAD BY DAY OF WEEK (0=Mon) ===")
print(df.groupby(df.index.dayofweek)[target].agg(["mean", "count"]).round(1))

print("\n=== AVERAGE LOAD BY MONTH ===")
print(df.groupby(df.index.month)[target].agg(["mean", "count"]).round(1))

print("\n=== AVERAGE LOAD BY YEAR ===")
print(df.groupby(df.index.year)[target].agg(["mean", "count"]).round(1))

print("\n=== AVERAGE LOAD BY HolidayFlag ===")
print(df.groupby("HolidayFlag")[target].agg(["mean", "count"]).round(1))

print("\n=== IS DayOfWeek COLUMN CONSISTENT WITH THE TIMESTAMP? ===")
print("Share of rows where DayOfWeek == weekday+1:",
      (df["DayOfWeek"] == df.index.dayofweek + 1).mean().round(4))

print("\n=== TIME-BASED AUTOCORRELATION (only pairs where both hours exist) ===")
full = df[target].asfreq("h")
for lag in [1, 2, 3, 24, 48, 168]:
    pair = pd.concat([full, full.shift(lag)], axis=1).dropna()
    print(f"lag {lag:>3}h: correlation = {pair.iloc[:, 0].corr(pair.iloc[:, 1]):+.3f}  (pairs = {len(pair)})")
