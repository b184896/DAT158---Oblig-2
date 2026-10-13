import pandas as pd
from datetime import datetime

# Sett Pandas-visning til å unngå vitenskapelig notasjon og vise 2 desimaler
pd.options.display.float_format = "{:,.2f}".format

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Les inn hele datasettet
df = pd.read_csv("data/vehicles.csv")
df = df.sample(n=20000, random_state=42).reset_index(drop=True)

target = "price"

# 1. UNDERSØK HELE DATASETTET FØR FILTERING
print("\nPrisfordeling i HELE datasettet før filtrering:")
print(df[target].describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.995]))

# 2. FJERN UTELIGGERE
df = df[df[target].between(500, 100_000)].copy()

# Feature engineering: Beregn bilens alder dynamisk
current_year = datetime.now().year
df["age"] = current_year - df["year"]

features = [
    "age",
    "region",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
]

x = df[features]
y = df[target]

# Splitte datasettet i trenings- og testsett FØR frekvenskoding for å unngå datalekkasje
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Frekvenskoding for kolonner med høy kardinalitet
high_cardinality_cols = ["model", "region", "manufacturer"]
for col in high_cardinality_cols:
    freq_map = x_train[col].value_counts().to_dict()
    x_train[col] = x_train[col].map(freq_map).fillna(0)
    x_test[col] = x_test[col].map(freq_map).fillna(0)

# Preprosesseringspipeline
numerical_features = ["age", "odometer"]
categorical_features = [col for col in features if col not in numerical_features]

numerical_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features),
])

# Bytte til HistGradientBoostingRegressor
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(
        random_state=42
    )),
])

print(f"\nTrener HistGradientBoostingRegressor på n = {len(df):,} annonser...")
model.fit(x_train, y_train)

predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print(f"\nTotalt antall annonser brukt etter filtrering: {len(df):,}")
print(f"MAE: {mae:,.0f} USD")
print(f"RMSE: {rmse:,.0f} USD")

example = x_test.iloc[[0]]
prediction = model.predict(example)[0]

print("\nEksempel på prisestimat:")
print(example.T.to_string(header=False))
print(f"Estimert pris: {prediction:,.0f} USD")
print(f"Faktisk annonsepris: {y_test.iloc[0]:,.0f} USD")
print(f"Differanse: {prediction - y_test.iloc[0]:,.0f} USD")
print(f"Prosentvis differanse: {(prediction - y_test.iloc[0]) / y_test.iloc[0] * 100:.2f}%")