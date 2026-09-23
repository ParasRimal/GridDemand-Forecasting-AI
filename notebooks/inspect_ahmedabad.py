import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("data/raw/weather and load dataset.csv")
target = "Electric Load (MW)"

print("=== SHAPE (rows, columns) ===")
print(df.shape)

print("\n=== COLUMN NAMES (exact) ===")
print(list(df.columns))

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES PER COLUMN ===")
print(df.isna().sum())

# Build one proper timestamp from the four separate columns
df["Timestamp"] = pd.to_datetime(
    dict(year=df["YEAR"], month=df["Month"], day=df["Day"], hour=df["Hour"])
)

print("\n=== TIMESTAMP CHECK ===")
print("Duplicate timestamps:", df["Timestamp"].duplicated().sum())
print("Sorted oldest to newest:", df["Timestamp"].is_monotonic_increasing)
print("First timestamp:", df["Timestamp"].min())
print("Last timestamp: ", df["Timestamp"].max())

print("\n=== GAPS BETWEEN CONSECUTIVE ROWS ===")
print(df["Timestamp"].diff().dropna().value_counts().head(10))

expected = pd.date_range(df["Timestamp"].min(), df["Timestamp"].max(), freq="h")
missing = expected.difference(df["Timestamp"])
print("\nExpected hourly timestamps:", len(expected))
print("Actual rows:               ", len(df))
print("Missing hours:             ", len(missing))
if len(missing) > 0:
    print(missing[:20])

print("\n=== TARGET SANITY ===")
print("Load <= 0:", (df[target] <= 0).sum())

print("\n=== SUMMARY STATISTICS ===")
print(df.drop(columns=["Timestamp"]).describe().T)
