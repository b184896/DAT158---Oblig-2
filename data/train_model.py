import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Les inn fra data-mappen med riktig filnavn
df = pd.read_csv("data/vehicles.csv")

# Definer funksjoner for å lage og evaluere modellen
features = [
    "year", 
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

# Definer target-variabelen
target = "price"

# Filtrer datasettet for å fjerne annonser med ekstreme priser
df = df[df[target].between(500, 100_000)].copy()

# Undersøk prisfordelingen før vi fjerner outliers
print("\nPrisfordeling før filtrering:")
print(df["price"].describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.995]
))

print("\nAntall annonser over ulike prisgrenser:")
for grense in [50_000, 100_000, 200_000, 500_000, 1_000_000]:
    antall = (df["price"] > grense).sum()
    print(f"Over {grense:>9,} USD: {antall:>7,} annonser")

df = df.sample(n=min(10_000, len(df)), random_state=42)

x = df[features]
y = df[target]

# Splitte datasettet i trenings- og testsett
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprosesseringspipeline for numeriske og kategoriske funksjoner
numerical_features = ["year", "odometer"]
categorical_features = [col for col in features if col not in numerical_features]

# Preprosesseringspipeline for numeriske funksjoner
numerical_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")),])

# Preprosesseringspipeline for kategoriske funksjoner
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),])

# Preprosesseringspipeline som kombinerer numeriske og kategoriske transformasjoner
preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features),])

# Lag en pipeline som inkluderer preprosesseringssteget og regresjonsmodellen
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=10, 
                                        random_state=42,
                                        n_jobs=-1,
                                        min_samples_leaf=2)),
    ])

print("Trener modellen...")
model.fit(x_train, y_train)

predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print(f"\nAntall annonser brukt: {len(df)}")
print(f"MAE: {mae:,.0f} USD")
print(f"RMSE: {rmse:,.0f} USD")

example = x_test.iloc[[0]]
prediction = model.predict(example)[0]

print("\nEksempel på prisestimat:")
print(example.to_string(index=False))
print(f"Estimert pris: {prediction:,.0f} USD")
print(f"Faktisk annonsepris: {y_test.iloc[0]:,.0f} USD")

'''
print("Første 5 rader i datasettet:")
print(df.head())

print("\nKolonner:")
print(df.columns.tolist())

print("\nInformasjon om datasettet:")
print(df.info())

print("\nManglende verdier per kolonne:")
print(df.isnull().sum())

print("Form på datasettet:", df.shape)
'''

