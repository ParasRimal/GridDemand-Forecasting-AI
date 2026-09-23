import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("data/raw/weather and load dataset.csv")
df["Timestamp"] = pd.to_datetime(
    dict(year=df["YEAR"], month=df["Month"], day=df["Day"], hour=df["Hour"])
)
target = "Electric Load (MW)"

print("=== IRRADIANCE BELOW ZERO ===")
neg_irr = df[df["irradiance"] < 0]
print("Rows with irradiance < 0:", len(neg_irr))
print("Distinct values:", sorted(neg_irr["irradiance"].unique()))
print(neg_irr[["Timestamp", "irradiance", target]].head(10))

print("\n=== LOAD <= 0 ===")
bad = df[target] <= 0
print("Rows with load <= 0:", bad.sum())
print(df.loc[bad, target].describe())

# Group consecutive bad hours into runs
run_id = (bad != bad.shift()).cumsum()
runs = df[bad].groupby(run_id[bad])["Timestamp"].agg(["min", "max", "count"])
print("\nNumber of separate runs:", len(runs))
print(runs.head(20))

print("\n=== LOAD BETWEEN 0 AND 10 (very low but positive) ===")
print(((df[target] > 0) & (df[target] < 10)).sum())

print("\n=== 10 LARGEST LOADS ===")
print(df.nlargest(10, target)[["Timestamp", target, "temperature"]])

print("\n=== LOAD PERCENTILES ===")
print(df[target].quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
