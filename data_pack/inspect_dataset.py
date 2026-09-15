import pandas as pd

file_path = "./archive/twcs/twcs.csv"
# Read only the first 1000 rows
df = pd.read_csv(file_path, nrows=1000)

print("\n========== DATASET COLUMNS ==========")
for column in df.columns:
    print(column)

print("\n========== DATASET SHAPE (SAMPLE) ==========")
print(df.shape)

print("\n========== FIRST 5 ROWS ==========")
print(df.head().to_string())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes)