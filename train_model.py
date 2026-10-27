import pandas as pd

df = pd.read_csv("data/vehicles.csv", nrows=100_000)

print(df.head())
print("Form på datasettet:", df.shape)
print(df.columns.tolist())