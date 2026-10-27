from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "vehicles.csv"
CSV_COLUMNS = [
    "price",
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
    "type",
    "state",
    "region",
]
MIN_YEAR = 1985
MAX_YEAR = 2022
MIN_ODOMETER = 1_000
MAX_ODOMETER = 300_000


def normalize(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text or "unknown"


def cylinders_to_number(value: Any) -> float | None:
    number = pd.to_numeric(
        pd.Series([str(value)]).str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    ).iloc[0]
    return None if pd.isna(number) else float(number)


def validate_vehicle(vehicle: dict[str, Any], data_path: Path = DATA_PATH) -> dict[str, Any]:
    """Compare one vehicle with the training data without loading the full CSV."""
    required = ["year", "odometer", "manufacturer", "model"]
    missing = [field for field in required if not str(vehicle.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Mangler obligatoriske felt: {', '.join(missing)}")

    input_year = int(vehicle["year"])
    input_odometer = int(vehicle["odometer"])
    input_values = {
        key: normalize(value)
        for key, value in vehicle.items()
        if key not in {"year", "odometer", "cylinders", "vehicle_type"}
    }
    input_values["type"] = normalize(vehicle.get("vehicle_type", vehicle.get("type")))
    input_values["cylinders"] = str(int(vehicle.get("cylinders", 4)))

    categorical_fields = [
        "manufacturer",
        "model",
        "condition",
        "fuel",
        "title_status",
        "transmission",
        "drive",
        "type",
        "state",
        "region",
    ]
    category_matches = {field: False for field in categorical_fields}
    similar: list[dict[str, Any]] = []
    exact_model_rows = 0

    usecols = CSV_COLUMNS
    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=100_000):
        for field in categorical_fields:
            values = chunk[field].map(normalize)
            category_matches[field] |= input_values[field] in set(values)

        chunk["manufacturer_normalized"] = chunk["manufacturer"].map(normalize)
        chunk["model_normalized"] = chunk["model"].map(normalize)
        model_rows = chunk[
            (chunk["manufacturer_normalized"] == input_values["manufacturer"])
            & (chunk["model_normalized"] == input_values["model"])
        ].copy()
        if model_rows.empty:
            continue
        model_rows["year"] = pd.to_numeric(model_rows["year"], errors="coerce")
        model_rows["odometer"] = pd.to_numeric(
            model_rows["odometer"], errors="coerce"
        )
        model_rows["price"] = pd.to_numeric(model_rows["price"], errors="coerce")
        model_rows = model_rows.dropna(subset=["year", "odometer", "price"])
        model_rows = model_rows[
            model_rows["price"].between(500, 120_000)
            & model_rows["year"].between(MIN_YEAR, MAX_YEAR)
            & model_rows["odometer"].between(MIN_ODOMETER, MAX_ODOMETER)
        ]
        exact_model_rows += len(model_rows)
        model_rows["distance"] = (
            (model_rows["year"] - input_year).abs()
            + (model_rows["odometer"] - input_odometer).abs() / 10_000
        )
        similar.extend(
            model_rows.nsmallest(10, "distance")[
                ["price", "year", "odometer", "condition", "state", "distance"]
            ].to_dict("records")
        )

    similar = sorted(similar, key=lambda row: row["distance"])[:5]
    return {
        "within_model_range": (
            MIN_YEAR <= input_year <= MAX_YEAR
            and MIN_ODOMETER <= input_odometer <= MAX_ODOMETER
        ),
        "category_matches": category_matches,
        "exact_manufacturer_model_rows": exact_model_rows,
        "similar_listings": similar,
    }


def print_report(vehicle: dict[str, Any], report: dict[str, Any]) -> None:
    print("\nDatavalidering mot vehicles.csv")
    print("=" * 36)
    print(
        f"Input: {vehicle['manufacturer']} {vehicle['model']}, "
        f"{vehicle['year']}, {vehicle['odometer']:,} miles"
    )
    print(
        "Innenfor modellens treningsområde: "
        + ("JA" if report["within_model_range"] else "NEI")
    )
    print(
        f"Gyldige annonser for produsent + modell: "
        f"{report['exact_manufacturer_model_rows']:,}"
    )
    print("\nFinnes verdien i datasettet?")
    for field, found in report["category_matches"].items():
        print(f"  {field:15} {'JA' if found else 'NEI'}")

    print("\nNærmeste annonser med samme produsent og modell:")
    if not report["similar_listings"]:
        print("  Ingen treff.")
        return
    for row in report["similar_listings"]:
        print(
            f"  ${row['price']:,.0f} | {row['year']:.0f} | "
            f"{row['odometer']:,.0f} miles | {normalize(row['condition'])} | "
            f"{normalize(row['state'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sjekk bilinput mot data/vehicles.csv."
    )
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--odometer", type=int, default=45_000)
    parser.add_argument("--manufacturer", default="Ford")
    parser.add_argument("--model", default="Mustang")
    parser.add_argument("--cylinders", type=int, default=6)
    parser.add_argument("--condition", default="Excellent")
    parser.add_argument("--fuel", default="gas")
    parser.add_argument("--title-status", dest="title_status", default="clean")
    parser.add_argument("--transmission", default="automatic")
    parser.add_argument("--drive", default="rwd")
    parser.add_argument("--type", dest="vehicle_type", default="sedan")
    parser.add_argument("--state", default="tx")
    parser.add_argument("--region", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = vars(parse_args())
    args["region"] = args["region"] or "unknown"
    report = validate_vehicle(args)
    print_report(args, report)
