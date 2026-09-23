import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("data/raw/weather and load dataset.csv")
df["Timestamp"] = pd.to_datetime(
    dict(year=df["YEAR"], month=df["Month"], day=df["Day"], hour=df["Hour"])
)
df = df.set_index("Timestamp").drop(columns=["YEAR", "Month", "Day", "Hour"])
target = "Electric Load (MW)"

# Mark bad values as missing IN MEMORY ONLY (the CSV file is not changed)
df.loc[df["irradiance"] == -999, "irradiance"] = float("nan")
df.loc[df[target] <= 0, target] = float("nan")

print("=== BAD IRRADIANCE BLOCK ===")
bad_irr = df.index[df["irradiance"].isna()]
print("Rows:", len(bad_irr), "| from", bad_irr.min(), "to", bad_irr.max())
print("One single block?", (bad_irr.max() - bad_irr.min()) == pd.Timedelta(hours=len(bad_irr) - 1))

print("\n=== BAD LOAD RUNS (last 4 of them) ===")
bad = df[target].isna()
run_id = (bad != bad.shift()).cumsum()
runs = df[bad].index.to_series().groupby(run_id[bad]).agg(["min", "max", "count"])
print("Total runs:", len(runs))
print(runs.tail(4))

print("\n=== CORRELATION OF EACH COLUMN WITH LOAD ===")
print(df.corr()[target].drop(target).sort_values().round(3))

print("\n=== AVERAGE LOAD BY HOUR OF DAY ===")
print(df.groupby(df.index.hour)[target].agg(["mean", "std", "count"]).round(1))

print("\n=== AVERAGE LOAD BY DAY OF WEEK (0=Mon) ===")
print(df.groupby(df.index.dayofweek)[target].agg(["mean", "count"]).round(1))

print("\n=== AVERAGE LOAD BY MONTH ===")
print(df.groupby(df.index.month)[target].agg(["mean", "count"]).round(1))

print("\n=== AUTOCORRELATION (row shift = hour shift, since no hours are missing) ===")
for lag in [1, 2, 3, 24, 48, 168]:
    pair = pd.concat([df[target], df[target].shift(lag)], axis=1).dropna()
    print(f"lag {lag:>3}h: correlation = {pair.iloc[:, 0].corr(pair.iloc[:, 1]):+.3f}  (pairs = {len(pair)})")
