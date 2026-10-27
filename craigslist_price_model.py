

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder


RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
PRICE_MIN = 500 #Unngå gis bort biler, og veldig billige biler som er urealistiske.
# Full-dataset describe(): 99.9% of prices is about $120,000.
PRICE_MAX = 120_000
ODO_MIN = 1_000
ODOMETER_MAX = 300_000
# Full-dataset describe(): 2.5% of years is about 1986.
MIN_YEAR = 1985

NUMERIC_FEATURES = [
    "year",
    "age",
    "odometer",
    "cylinders_num",
]
HIGH_CARDINALITY_FEATURES = ["manufacturer", "model", "region"]
CATEGORICAL_FEATURES = [
    "condition",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "state",
]
FEATURES = NUMERIC_FEATURES + HIGH_CARDINALITY_FEATURES + CATEGORICAL_FEATURES
REQUIRED_FEATURES = ["model", "year", "odometer"]


def load_and_clean(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    """Load the CSV and remove observations that cannot represent a normal ad."""
    df = pd.read_csv(path)
    loaded = len(df)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=RANDOM_STATE)
    selected = len(df)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    df["model"] = df["model"].astype("string").str.strip()
    df.loc[df["model"].eq(""), "model"] = pd.NA
    missing_required = df[REQUIRED_FEATURES].isna().sum()
    print("\nMissing required features before filtering:")
    print(missing_required.to_string())
    df = df.dropna(subset=REQUIRED_FEATURES).copy()

    zero_odometer = df["odometer"].eq(0)
    print(
        f"Original zero-odometer rows: {int(zero_odometer.sum()):,}; "
        f"their year range: "
        f"{df.loc[zero_odometer, 'year'].min():.0f}-"
        f"{df.loc[zero_odometer, 'year'].max():.0f}"
    )
    df["cylinders_num"] = (
        df["cylinders"].astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0]
    )
    df["cylinders_num"] = pd.to_numeric(df["cylinders_num"], errors="coerce")

    posting_date = pd.to_datetime(df["posting_date"], errors="coerce", utc=True)
    df["listing_year"] = posting_date.dt.year
    df["listing_month"] = posting_date.dt.month
    # A missing posting date is uncommon; the median is a neutral fallback.
    df["listing_year"] = df["listing_year"].fillna(2021)
    df["listing_month"] = df["listing_month"].fillna(6)
    df["age"] = df["listing_year"] - df["year"]

    # Missing categories are real information here, so preserve them explicitly.
    for column in HIGH_CARDINALITY_FEATURES + CATEGORICAL_FEATURES:
        df[column] = df[column].astype("string").fillna("unknown")

    describe = df[["price", "year", "odometer"]].describe(
        percentiles=[0.001, 0.005, 0.01, 0.025, 0.5, 0.975, 0.99, 0.995, 0.999]
    )
    print("\nDescribe før filtrering:")
    print(describe.to_string())

    df = df[
        df["price"].between(PRICE_MIN, PRICE_MAX)
        & df["year"].between(MIN_YEAR, df["listing_year"] + 1)
        & df["odometer"].between(ODO_MIN, ODOMETER_MAX)
        & df["age"].between(0, 100)
    ].copy()

    # The same ad can occur more than once. Keep one row per stable ad identity.
    duplicate_columns = [
        column
        for column in ["url", "price", "year", "manufacturer", "model", "odometer"]
        if column in df
    ]
    if duplicate_columns:
        df = df.drop_duplicates(subset=duplicate_columns)

    print(
        f"Loaded {loaded:,} rows"
        f"{f' and sampled {selected:,}' if selected != loaded else ''}; "
        f"removed {selected - len(df):,} rows during "
        f"filtering/deduplication; using {len(df):,} rows."
    )
    print(
        f"Price range kept: ${df['price'].min():,.0f}-${df['price'].max():,.0f}; "
        f"total rows removed including sampling: {loaded - len(df):,}"
    )
    print(f"Required features: {', '.join(REQUIRED_FEATURES)}")
    print(f"Odometer range kept: {ODO_MIN:,}-{ODOMETER_MAX:,} miles")
    return df.reset_index(drop=True)


def build_model() -> Pipeline:
    """Build the selected Extra Trees model and its preprocessing pipeline."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    high_cardinality = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "target_encoder",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=5,
                ),
            ),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("high_cardinality", high_cardinality, HIGH_CARDINALITY_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_overfitting_underfitting(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Train on train data and compare performance on train, validation and test."""
    model.fit(x_train, y_train)

    datasets = [
        ("Train", x_train, y_train),
        ("Validation", x_validation, y_validation),
        ("Test", x_test, y_test),
    ]
    print("\nOverfitting/underfitting diagnostic:")
    print(f"{'Dataset':<12}{'Rows':>10}{'MAE':>14}{'RMSE':>14}{'MAE % of mean':>18}")
    for name, features, target in datasets:
        predictions = model.predict(features)
        mae = mean_absolute_error(target, predictions)
        rmse = root_mean_squared_error(target, predictions)
        mae_percentage = 100 * mae / target.mean()
        print(
            f"{name:<12}{len(target):>10,}"
            f"{mae:>14,.0f}{rmse:>14,.0f}{mae_percentage:>17.1f}%"
        )


# def print_importance(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
#     """Report importance by original input column, avoiding one-hot column noise."""
#     result = permutation_importance(
#         model,
#         x_test,
#         y_test,
#         scoring="neg_mean_absolute_error",
#         n_repeats=3,
#         random_state=RANDOM_STATE,
#         n_jobs=-1,
#     )
#     importance = pd.Series(result.importances_mean, index=x_test.columns).sort_values(
#         ascending=False
#     )
#     print("\nFeature importance (mean increase in MAE when shuffled):")
#     for feature, value in importance.items():
#         print(f"  {feature:16s} {value:,.0f} USD")


def main(path: Path, sample_size: int | None = None) -> None:
    df = load_and_clean(path, sample_size)
    x = df[FEATURES]
    y = df["price"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    print(
        f"\nTraining on {len(x_train):,} rows, validating on "
        f"{len(x_validation):,} rows and testing on {len(x_test):,} rows..."
    )

    print("\nFiltered target distribution:")
    print(y.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    evaluate_overfitting_underfitting(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    )
    # print_importance(model, x_test, y_test)


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "vehicles.csv"
    requested_sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(csv_path, requested_sample)
