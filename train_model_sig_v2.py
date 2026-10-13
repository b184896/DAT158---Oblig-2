import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
)
 
pd.options.display.float_format = "{:,.2f}".format
RANDOM_STATE = 42
 
# ---------------------------------------------------------------------------
# 1. Les inn HELE datasettet (ingen nedsampling)
# ---------------------------------------------------------------------------
df = pd.read_csv("data/vehicles.csv")
print(f"Rader totalt: {len(df):,}")
 
target = "price"
 
# ---------------------------------------------------------------------------
# 2. Rensing
# ---------------------------------------------------------------------------
# Pris
df = df[df[target].between(500, 100_000)].copy()
 
# Alder ved annonsetidspunkt (mer riktig enn "dagens år")
df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce", utc=True)
df["posting_year"] = df["posting_date"].dt.year
df["posting_month"] = df["posting_date"].dt.month
df["age"] = df["posting_year"] - df["year"]
df = df[df["age"].between(0, 35)]  # fjerner manglende/urealistiske årstall
 
# Kilometerstand: behold NaN (imputeres av modellen), fjern urealistiske verdier
df = df[df["odometer"].isna() | df["odometer"].between(1_000, 400_000)]
 
# Koordinater: sett utenfor USA til NaN
df.loc[~df["lat"].between(24, 50), ["lat", "long"]] = np.nan
df.loc[~df["long"].between(-125, -66), ["lat", "long"]] = np.nan
 
# Sylindere som tall ("4 cylinders" -> 4, "other" -> NaN)
df["cylinders_num"] = pd.to_numeric(
    df["cylinders"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
)
 
# Normaliser modellnavn
df["model"] = df["model"].astype(str).str.lower().str.strip()
 
# ---------------------------------------------------------------------------
# 3. Fjern duplikater FØR splitt (hindrer lekkasje mellom train og test)
# ---------------------------------------------------------------------------
n_before = len(df)
vin_col = "VIN" if "VIN" in df.columns else ("vin" if "vin" in df.columns else None)
if vin_col:
    has_vin = df[vin_col].notna()
    df = df[~(has_vin & df.duplicated(subset=[vin_col], keep="first"))]
df = df.drop_duplicates(
    subset=["manufacturer", "model", "year", "odometer", "price"], keep="first"
)
print(f"Duplikater fjernet: {n_before - len(df):,}  ->  {len(df):,} annonser igjen")
 
# ---------------------------------------------------------------------------
# 4. Features
# ---------------------------------------------------------------------------
numerical_features = ["age", "odometer", "cylinders_num", "lat", "long", "posting_month"]
high_card_features = ["model", "region", "manufacturer", "state"]  # TargetEncoder
low_card_features = [
    "condition", "fuel", "title_status", "transmission",
    "drive", "size", "type", "paint_color",
]  # One-hot
categorical_features = high_card_features + low_card_features
 
features = numerical_features + categorical_features
 
for col in categorical_features:
    df[col] = df[col].fillna("missing").astype(str)
 
x = df[features]
y = df[target]
 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {len(x_train):,}  Test: {len(x_test):,}")
 
# ---------------------------------------------------------------------------
# 5. Modell: TargetEncoder + HistGradientBoosting + log-transform av målet
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    [
        ("num", "passthrough", numerical_features),  # HistGB håndterer NaN selv
        ("hi", TargetEncoder(), high_card_features),
        (
            "lo",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=50,
                sparse_output=False,
            ),
            low_card_features,
        ),
    ]
)
 
model = TransformedTargetRegressor(
    regressor=Pipeline(
        [
            ("prep", preprocessor),
            ("gb", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    ),
    func=np.log1p,
    inverse_func=np.expm1,
)
 
# ---------------------------------------------------------------------------
# 6. Hyperparametertuning (kun på treningsdata, CV scorer i USD)
# ---------------------------------------------------------------------------
param_distributions = {
    "regressor__gb__learning_rate": [0.03, 0.05, 0.1, 0.15],
    "regressor__gb__max_iter": [300, 500, 800],
    "regressor__gb__max_leaf_nodes": [31, 63, 127],
    "regressor__gb__min_samples_leaf": [20, 50, 100],
    "regressor__gb__l2_regularization": [0.0, 0.1, 1.0, 10.0],
}
 
search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)
 
print("\nTuner modellen med kryssvalidering på treningsdata...")
search.fit(x_train, y_train)
print(f"Beste parametere: {search.best_params_}")
print(f"Beste CV MAE: {-search.best_score_:,.0f} USD")
 
best_model = search.best_estimator_  # refittet på hele train
 
# ---------------------------------------------------------------------------
# 7. Baseline: median pris per manufacturer + aldersgruppe
# ---------------------------------------------------------------------------
def age_bin(a):
    return pd.cut(a, bins=[-1, 2, 5, 10, 15, 35], labels=False)
 
base_train = pd.DataFrame(
    {
        "manufacturer": x_train["manufacturer"],
        "age_bin": age_bin(x_train["age"]),
        "price": y_train,
    }
)
medians = base_train.groupby(["manufacturer", "age_bin"])["price"].median()
global_median = y_train.median()
 
base_keys = pd.MultiIndex.from_arrays(
    [x_test["manufacturer"], age_bin(x_test["age"])]
)
baseline_pred = pd.Series(medians.reindex(base_keys).values, index=x_test.index).fillna(
    global_median
)
 
# ---------------------------------------------------------------------------
# 8. Evaluering på testsett (brukes kun én gang)
# ---------------------------------------------------------------------------
def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    medape = np.median(np.abs(y_pred - y_true) / y_true) * 100
    print(
        f"{name:<10} MAE: {mae:>7,.0f} | MedAE: {medae:>7,.0f} | "
        f"RMSE: {rmse:>7,.0f} | MAPE: {mape:5.1f}% | MedAPE: {medape:5.1f}%"
    )
 
predictions = best_model.predict(x_test)
 
print("\nResultat på testsett (USD):")
report("Baseline", y_test, baseline_pred)
report("GradBoost", y_test, predictions)
 
# ---------------------------------------------------------------------------
# 9. Eksempel
# ---------------------------------------------------------------------------
example = x_test.iloc[[0]]
prediction = best_model.predict(example)[0]
actual = y_test.iloc[0]
 
print("\nEksempel på prisestimat:")
print(example.T.to_string(header=False))
print(f"Estimert pris: {prediction:,.0f} USD")
print(f"Faktisk annonsepris: {actual:,.0f} USD")
print(f"Differanse: {prediction - actual:,.0f} USD")
print(f"Prosentvis differanse: {(prediction - actual) / actual * 100:.2f}%")