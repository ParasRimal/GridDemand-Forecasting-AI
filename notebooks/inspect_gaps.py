import pandas as pd

pd.set_option("display.width", 200)

df = pd.read_csv("data/raw/PowerLoad_Dataset.csv", parse_dates=["Timestamp"])
ts = df["Timestamp"]
gaps = ts.diff().dropna()
gap_hours = gaps.dt.total_seconds() / 3600

print("=== GAP BETWEEN CONSECUTIVE ROWS (in hours) ===")
print(gap_hours.describe())

print("\n=== 10 MOST COMMON GAPS ===")
print(gaps.value_counts().head(10))

print("\n=== 5 LARGEST GAPS ===")
print(gaps.nlargest(5))

print("\n=== HOW MANY GAPS ARE EXACTLY 1 HOUR? ===")
print((gaps == pd.Timedelta(hours=1)).sum(), "out of", len(gaps))

print("\n=== ROWS PER CALENDAR DAY ===")
rows_per_day = ts.dt.normalize().value_counts()
print("Distinct days with data:", len(rows_per_day))
print(rows_per_day.describe())

print("\n=== ROWS PER HOUR OF DAY ===")
print(ts.dt.hour.value_counts().sort_index())

print("\n=== TIMESTAMPS NOT ON THE EXACT HOUR ===")
print((ts.dt.minute != 0).sum())
