import pandas as pd
import matplotlib.pyplot as plt

# Les inn datasettet
df = pd.read_csv("data/vehicles.csv")

# Fjern annonser uten gyldig pris
df = df[df["price"] > 0].copy()


# --------------------------------------------------
# 1. Fordeling av bruktbilpriser
# --------------------------------------------------

# Avgrens prisene for å gjøre grafen mer lesbar
prices = df.loc[df["price"].between(500, 100_000), "price"]

plt.figure(figsize=(10, 5))
plt.hist(
    prices,
    bins=60,
    log=True,
    edgecolor="white"
)

plt.title("Fordeling av bruktbilpriser")
plt.xlabel("Annonsepris (USD)")
plt.ylabel("Antall annonser (logaritmisk skala)")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Sammenheng mellom kjørelengde og pris
# --------------------------------------------------

# Filtrer verdier for å unngå at ekstreme verdier
# gjør grafen vanskelig å lese
filtered_df = df[
    df["odometer"].between(0, 300_000)
    & df["price"].between(500, 100_000)
]

# Bruk et tilfeldig utvalg på maksimalt 3000 annonser
sample = filtered_df.sample(
    n=min(3000, len(filtered_df)),
    random_state=42
)

plt.figure(figsize=(10, 5))
plt.scatter(
    sample["odometer"],
    sample["price"],
    s=8,
    alpha=0.15
)

plt.title("Sammenheng mellom kjørelengde og pris")
plt.xlabel("Kjørelengde (miles)")
plt.ylabel("Annonsepris (USD)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Medianpris for de 10 vanligste bilmerkene
# --------------------------------------------------

top_manufacturers = (
    df["manufacturer"]
    .dropna()
    .value_counts()
    .head(10)
    .index
)

manufacturer_prices = (
    df[df["manufacturer"].isin(top_manufacturers)]
    .groupby("manufacturer")["price"]
    .median()
    .sort_values()
)

plt.figure(figsize=(10, 6))
manufacturer_prices.plot(
    kind="barh",
    edgecolor="black"
)

plt.title("Medianpris for de 10 vanligste bilmerkene")
plt.xlabel("Medianpris (USD)")
plt.ylabel("Bilmerke")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
