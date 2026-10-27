from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "craigslist_price_model.joblib"
LISTING_YEAR = 2021
MIN_YEAR = 1985
MAX_YEAR = LISTING_YEAR + 1
MIN_ODOMETER = 1_000
MAX_ODOMETER = 300_000
MODEL_FEATURES = [
    "year",
    "age",
    "odometer",
    "cylinders_num",
    "manufacturer",
    "model",
    "region",
    "condition",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "state",
]


st.set_page_config(
    page_title="Prisestimat for bruktbil",
    page_icon="🚗",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Fant ikke modellfilen '{MODEL_PATH.name}' i {BASE_DIR}."
        )
    return joblib.load(MODEL_PATH)


def build_input_data(
    *,
    year: int,
    odometer: int,
    manufacturer: str,
    model_name: str,
    cylinders: int,
    condition: str,
    fuel: str,
    title_status: str,
    transmission: str,
    drive: str,
    vehicle_type: str,
    region: str,
    state: str,
) -> pd.DataFrame:
    """Create the exact feature table expected by the saved model."""
    manufacturer = manufacturer.strip().lower()
    model_name = model_name.strip().lower()
    if not manufacturer or not model_name:
        raise ValueError("Produsent og modell må fylles ut.")
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError(f"Årsmodell må være mellom {MIN_YEAR} og {MAX_YEAR}.")
    if not MIN_ODOMETER <= odometer <= MAX_ODOMETER:
        raise ValueError(
            f"Kilometerstand må være mellom {MIN_ODOMETER:,} og "
            f"{MAX_ODOMETER:,} miles."
        )

    return pd.DataFrame(
        [
            {
                "year": year,
                "age": LISTING_YEAR - year,
                "odometer": odometer,
                "cylinders_num": cylinders,
                "manufacturer": manufacturer,
                "model": model_name,
                "region": region.strip().lower() or "unknown",
                "condition": condition.strip().lower(),
                "fuel": fuel.strip().lower(),
                "title_status": title_status.strip().lower(),
                "transmission": transmission.strip().lower(),
                "drive": drive.strip().lower(),
                "type": vehicle_type.strip().lower(),
                "state": state.strip().lower() or "unknown",
            }
        ],
        columns=MODEL_FEATURES,
    )


def predict_price(model, input_data: pd.DataFrame) -> float:
    """Run one prediction and reject an invalid model result."""
    prediction = float(model.predict(input_data)[0])
    if prediction <= 0:
        raise ValueError("Modellen returnerte en ugyldig, ikke-positiv pris.")
    return prediction


try:
    model = load_model()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Kunne ikke laste prismodellen: {error}")
    st.stop()


st.title("🚗 Prisestimat for bruktbil")
st.write(
    "Fyll inn bilens egenskaper for å få et raskt estimat basert på "
    "historiske Craigslist-annonser."
)
st.info(
    "Estimater gjelder biler innenfor modellens treningsområde "
    f"({MIN_YEAR}–{MAX_YEAR}, {MIN_ODOMETER:,}–{MAX_ODOMETER:,} miles). "
    "Resultatet er veiledende og ikke en garantert salgspris."
)


with st.form("vehicle_form"):
    st.subheader("Bilopplysninger")
    left, right = st.columns(2)

    with left:
        year = st.number_input(
            "Årsmodell",
            min_value=MIN_YEAR,
            max_value=MAX_YEAR,
            value=2018,
            step=1,
        )
        odometer = st.number_input(
            "Kilometerstand (miles)",
            min_value=MIN_ODOMETER,
            max_value=MAX_ODOMETER,
            value=80_000,
            step=1_000,
        )
        manufacturer = st.text_input("Produsent", value="Toyota")
        model_name = st.text_input("Modell", value="Camry")
        cylinders = st.selectbox("Antall sylindre", [3, 4, 5, 6, 8, 10, 12], index=1)
        condition = st.selectbox(
            "Tilstand",
            ["good", "excellent", "like new", "fair", "new", "salvage", "unknown"],
        )
        fuel = st.selectbox(
            "Drivstoff",
            ["gas", "diesel", "hybrid", "electric", "other", "unknown"],
        )

    with right:
        title_status = st.selectbox(
            "Tittelstatus",
            ["clean", "rebuilt", "salvage", "lien", "missing", "parts only", "unknown"],
        )
        transmission = st.selectbox(
            "Girkasse",
            ["automatic", "manual", "other", "unknown"],
        )
        drive = st.selectbox("Drivlinje", ["4wd", "fwd", "rwd", "unknown"])
        vehicle_type = st.selectbox(
            "Biltype",
            [
                "sedan",
                "SUV",
                "pickup",
                "truck",
                "coupe",
                "hatchback",
                "wagon",
                "van",
                "convertible",
                "mini-van",
                "other",
                "unknown",
            ],
        )
        region = st.text_input(
            "Region",
            help="Skriv Craigslist-regionen hvis du kjenner den.",
        )
        state = st.text_input(
            "Delstat (to bokstaver)",
            value="ca",
            max_chars=2,
            help="For eksempel ca, tx eller ny.",
        )

    submitted = st.form_submit_button(
        "Beregn prisestimat",
        type="primary",
        use_container_width=True,
    )


if submitted:
    try:
        input_data = build_input_data(
            year=int(year),
            odometer=int(odometer),
            manufacturer=manufacturer,
            model_name=model_name,
            cylinders=int(cylinders),
            condition=condition,
            fuel=fuel,
            title_status=title_status,
            transmission=transmission,
            drive=drive,
            vehicle_type=vehicle_type,
            region=region,
            state=state,
        )
        prediction = predict_price(model, input_data)
    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Kunne ikke beregne prisestimatet: {error}")
    else:
        st.success("Prisestimatet er klart")
        st.metric("Estimert annonsepris", f"${prediction:,.0f} USD")
        st.caption(
            "Faktisk pris kan variere med blant annet servicehistorikk, "
            "utstyr, geografisk marked og markedsendringer."
        )


st.divider()
st.caption("DAT158 — Prisestimat for bruktbil | Extra Trees Regressor")
