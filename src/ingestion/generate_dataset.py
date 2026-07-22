"""
Generate a realistic hotel-booking dataset that mirrors the schema and the
'messiness' of the well-known Hotel Booking Demand dataset (Antonio, Almeida &
Nunes, 2019). We deliberately inject the same real-world data-quality problems
so that every cleaning technique in Stage 1 has something real to bite on:

  - missing values (children, country, agent, company)
  - impossible / invalid rows (adr < 0, zero-guest bookings)
  - duplicates
  - high-cardinality categoricals (country, agent, company)
  - a strong leakage trap (reservation_status <-> is_canceled)
  - class imbalance (~37% cancellations, like the real data)

This is a stand-in so the whole pipeline is runnable offline. Swap in the real
CSV later and every downstream step still works.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 119_390  # same row count as the real dataset

hotels = rng.choice(["Resort Hotel", "City Hotel"], size=N, p=[0.33, 0.67])

# Lead time is right-skewed: most bookings are near-term, a long tail books early.
lead_time = rng.gamma(shape=2.0, scale=52, size=N).astype(int)
lead_time = np.clip(lead_time, 0, 737)

arrival_year = rng.choice([2015, 2016, 2017], size=N, p=[0.25, 0.45, 0.30])
months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
# Seasonality: summer months get more bookings.
month_p = np.array([6, 6, 7, 8, 9, 10, 12, 12, 10, 8, 6, 6], dtype=float)
month_p /= month_p.sum()
arrival_month = rng.choice(months, size=N, p=month_p)

market_segment = rng.choice(
    ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"],
    size=N,
    p=[0.47, 0.20, 0.12, 0.11, 0.07, 0.02, 0.01],
)
deposit_type = rng.choice(["No Deposit", "Non Refund", "Refundable"], size=N, p=[0.88, 0.11, 0.01])
customer_type = rng.choice(
    ["Transient", "Transient-Party", "Contract", "Group"], size=N, p=[0.75, 0.21, 0.03, 0.01]
)

adults = rng.choice([1, 2, 3, 4], size=N, p=[0.20, 0.72, 0.06, 0.02])
children = rng.choice([0, 1, 2, 3], size=N, p=[0.92, 0.05, 0.025, 0.005]).astype(float)
babies = rng.choice([0, 1, 2], size=N, p=[0.99, 0.009, 0.001])

# ADR (average daily rate): depends on hotel type + a seasonal bump, plus noise.
base_adr = np.where(hotels == "City Hotel", 105, 95)
season_bump = np.isin(arrival_month, ["June", "July", "August"]).astype(float) * 40
adr = base_adr + season_bump + rng.normal(0, 35, size=N)

prev_cancellations = rng.choice([0, 1, 2, 3], size=N, p=[0.94, 0.04, 0.015, 0.005])
booking_changes = rng.choice([0, 1, 2, 3], size=N, p=[0.85, 0.10, 0.035, 0.015])
special_requests = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.58, 0.26, 0.11, 0.04, 0.01])
is_repeated_guest = rng.choice([0, 1], size=N, p=[0.968, 0.032])

# High-cardinality categoricals with realistic skew + heavy missingness.
countries = [
    "PRT",
    "GBR",
    "FRA",
    "ESP",
    "DEU",
    "ITA",
    "IRL",
    "BEL",
    "BRA",
    "NLD",
    "USA",
    "CHE",
    "AUT",
    "SWE",
    "CHN",
    "POL",
    "ISR",
    "RUS",
    "NOR",
    "OTHER",
]
country = rng.choice(
    countries,
    size=N,
    p=np.array([41, 10, 9, 7, 6, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 4]) / 100,
)
agent = rng.choice(
    [*range(1, 330), np.nan], size=N, p=np.concatenate([np.full(329, 0.86 / 329), [0.14]])
)
company = rng.choice(
    [*range(1, 350), np.nan], size=N, p=np.concatenate([np.full(349, 0.05 / 349), [0.95]])
)

# --- Build the cancellation label with a real signal, not pure noise ---
logit = (
    -1.15
    + 0.0045 * lead_time
    + 0.9 * (deposit_type == "Non Refund").astype(float)
    + 0.6 * (market_segment == "Groups").astype(float)
    + 0.5 * prev_cancellations
    - 0.35 * special_requests
    - 0.4 * is_repeated_guest
    - 0.25 * booking_changes
)
prob_cancel = 1 / (1 + np.exp(-logit))
is_canceled = (rng.random(N) < prob_cancel).astype(int)

# reservation_status is a LEAKAGE trap: it's determined post-outcome.
reservation_status = np.where(
    is_canceled == 1, rng.choice(["Canceled", "No-Show"], size=N, p=[0.9, 0.1]), "Check-Out"
)

df = pd.DataFrame(
    {
        "hotel": hotels,
        "is_canceled": is_canceled,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": arrival_month,
        "adults": adults,
        "children": children,
        "babies": babies,
        "market_segment": market_segment,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "adr": adr.round(2),
        "previous_cancellations": prev_cancellations,
        "booking_changes": booking_changes,
        "total_of_special_requests": special_requests,
        "is_repeated_guest": is_repeated_guest,
        "country": country,
        "agent": agent,
        "company": company,
        "reservation_status": reservation_status,
    }
)

# --- Inject real-world messiness ---
# 1. Missing children (4 rows in the real set) + missing country.
df.loc[rng.choice(N, 4, replace=False), "children"] = np.nan
df.loc[rng.choice(N, int(0.004 * N), replace=False), "country"] = np.nan
# 2. Impossible ADR (a famous outlier of 5400 exists in the real data) + negatives.
df.loc[rng.choice(N, 1, replace=False), "adr"] = 5400.0
df.loc[rng.choice(N, 20, replace=False), "adr"] = rng.uniform(-50, -1, 20).round(2)
# 3. Zero-guest bookings (adults=children=babies=0) — invalid.
zero_idx = rng.choice(N, 180, replace=False)
df.loc[zero_idx, ["adults", "children", "babies"]] = 0
# 4. Exact duplicate rows.
dups = df.sample(300, random_state=1)
df = pd.concat([df, dups], ignore_index=True)

df.to_csv("data/raw/hotel_bookings.csv", index=False)
print(f"Wrote {len(df):,} rows -> data/raw/hotel_bookings.csv")
print(f"Cancellation rate: {df.is_canceled.mean():.3f}")
print(f"Missing children: {df.children.isna().sum()}, missing country: {df.country.isna().sum()}")
print(f"Missing agent: {df.agent.isna().sum():,}, missing company: {df.company.isna().sum():,}")
