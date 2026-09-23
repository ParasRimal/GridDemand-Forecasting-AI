import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("data/raw/PowerLoad_Dataset.csv", parse_dates=["Timestamp"])

print("=== SHAPE (rows, columns) ===")
print(df.shape)

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES PER COLUMN ===")
print(df.isna().sum())

print("\n=== DUPLICATES ===")
print("Fully duplicated rows:", df.duplicated().sum())
print("Duplicate timestamps:", df["Timestamp"].duplicated().sum())

print("\n=== TIMESTAMP CHECK ===")
print("Sorted oldest to newest:", df["Timestamp"].is_monotonic_increasing)
print("First timestamp:", df["Timestamp"].min())
print("Last timestamp: ", df["Timestamp"].max())

print("\n=== SUMMARY STATISTICS ===")
print(df.describe().T)
