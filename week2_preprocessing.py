"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis

Reference dataset: the DataCo Smart Supply Chain dataset (Kaggle) describes shipments
with order/ship dates, shipping mode, region, sales value, planned vs actual shipping
days and delivery status. Because the raw file cannot be downloaded in every
environment, this script SIMULATES a shipment table with the same kind of fields and
injects the data-quality problems that real logistics extracts contain. Replace
`simulate_raw_shipments()` with `pd.read_csv(<path>, encoding="latin-1")` to run the
same pipeline on the real file (column names then need a rename map).

Pipeline: audit -> structural cleaning -> categorical standardisation -> validity rules
          -> missing values -> outliers -> feature engineering -> encoding -> scaling
          -> validation checks.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

rng = np.random.default_rng(7)
METRICS = {}


# ---------------------------------------------------------------------------
# 1. DATA COLLECTION (simulated)
# ---------------------------------------------------------------------------
def simulate_raw_shipments(n=6000):
    modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    planned = {"Standard Class": 5, "Second Class": 3, "First Class": 2, "Same Day": 1}
    regions = ["North", "South", "East", "West", "Central"]

    ship_mode = rng.choice(modes, n, p=[0.55, 0.2, 0.15, 0.10])
    region = rng.choice(regions, n)
    distance = np.clip(rng.lognormal(5.2, 0.9, n), 5, 1800)
    weight = np.clip(rng.gamma(2.2, 6.0, n), 0.5, 120)
    value = weight * rng.uniform(150, 450, n) + rng.normal(0, 400, n)
    value = np.clip(value, 100, None)
    plan = np.array([planned[m] for m in ship_mode])
    transit = np.maximum(1, np.round(plan + rng.normal(0.3, 1.1, n) + distance / 900)).astype(int)
    order_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    ship_date = order_date + pd.to_timedelta(rng.integers(0, 3, n), unit="D")
    delivery_date = ship_date + pd.to_timedelta(transit, unit="D")

    df = pd.DataFrame({
        "order_id": np.arange(100000, 100000 + n),
        "order_date": order_date, "ship_date": ship_date, "delivery_date": delivery_date,
        "ship_mode": ship_mode, "region": region, "distance_km": distance.round(1),
        "weight_kg": weight.round(2), "order_value_inr": value.round(2),
        "planned_days": plan, "stock_on_hand": rng.integers(0, 500, n).astype(float),
    })

    # ---- inject realistic data-quality problems ----
    # (a) mixed date formats: ~15 % of rows written as DD/MM/YYYY text
    for c in ["order_date", "ship_date", "delivery_date"]:
        df[c] = df[c].dt.strftime("%Y-%m-%d")
        idx = rng.choice(n, int(0.15 * n), replace=False)
        df.loc[idx, c] = pd.to_datetime(df.loc[idx, c]).dt.strftime("%d/%m/%Y")
    # (b) inconsistent categorical labels
    messy_mode = {"Standard Class": ["standard class", "Standard Class ", "STD CLASS"],
                  "Second Class": ["second class", "2nd Class"],
                  "First Class": ["first class", "FIRST CLASS "],
                  "Same Day": ["same day", "Same-Day"]}
    for k, v in messy_mode.items():
        idx = df.index[(df.ship_mode == k) & (rng.random(n) < 0.18)]
        df.loc[idx, "ship_mode"] = rng.choice(v, len(idx))
    idx = rng.choice(n, 300, replace=False)
    df.loc[idx, "region"] = df.loc[idx, "region"].str.lower().add(" ")
    # (c) numbers stored as text
    df["weight_kg"] = df["weight_kg"].astype(object)
    idx = rng.choice(n, 480, replace=False)
    df.loc[idx, "weight_kg"] = df.loc[idx, "weight_kg"].astype(str) + " kg"
    df.loc[rng.choice(n, 60, replace=False), "weight_kg"] = "N/A"
    # (d) missing values (some depend on shipping mode -> not purely random)
    for col, p in [("distance_km", 0.04), ("order_value_inr", 0.03), ("region", 0.02),
                   ("stock_on_hand", 0.05), ("delivery_date", 0.04)]:
        df.loc[rng.random(n) < p, col] = np.nan
    df.loc[(df.ship_mode.str.lower().str.contains("same")) & (rng.random(n) < 0.12), "weight_kg"] = np.nan
    # (e) impossible values
    df.loc[rng.choice(n, 15, replace=False), "distance_km"] = -1 * rng.uniform(10, 300, 15)
    df.loc[rng.choice(n, 10, replace=False), "weight_kg"] = 0
    # (f) delivery before shipment (swapped ISO dates)
    bad = rng.choice(n, 40, replace=False)
    df.loc[bad, "delivery_date"] = "2024-12-30"
    # (g) extreme entry errors
    df.loc[rng.choice(n, 25, replace=False), "weight_kg"] = rng.uniform(2000, 9000, 25).round(1)
    df.loc[rng.choice(n, 20, replace=False), "order_value_inr"] = rng.uniform(5e6, 2e7, 20).round(2)
    # (h) duplicates: exact copies and repeated order IDs
    dup_exact = df.sample(120, random_state=1)
    dup_id = df.sample(40, random_state=2).copy()
    dup_id["order_value_inr"] = dup_id["order_value_inr"].astype(float) * 1.0
    return pd.concat([df, dup_exact, dup_id], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. DATA AUDIT
# ---------------------------------------------------------------------------
def audit(df, label):
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str),
                        "missing": df.isna().sum(),
                        "missing_%": (100 * df.isna().mean()).round(2),
                        "unique": df.nunique()})
    print(f"\n--- AUDIT: {label} ({len(df)} rows) ---\n", rep)
    return rep


# ---------------------------------------------------------------------------
# 3. CLEANING STEPS
# ---------------------------------------------------------------------------
def remove_duplicates(df):
    n0 = len(df)
    df = df.drop_duplicates()
    n1 = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    METRICS["duplicates_exact"] = n0 - n1
    METRICS["duplicates_id"] = n1 - len(df)
    return df.reset_index(drop=True)


def fix_types(df):
    df = df.copy()
    w = df["weight_kg"].astype(str).str.replace("kg", "", regex=False).str.strip()
    w = w.replace({"N/A": np.nan, "nan": np.nan, "None": np.nan})
    df["weight_kg"] = pd.to_numeric(w, errors="coerce")
    for c in ["order_date", "ship_date", "delivery_date"]:
        iso = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
        dmy = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
        df[c] = iso.fillna(dmy)                       # try ISO first, fall back to DD/MM/YYYY
    return df


def standardise_categories(df):
    df = df.copy()
    mode_map = {"standard class": "Standard Class", "std class": "Standard Class",
                "second class": "Second Class", "2nd class": "Second Class",
                "first class": "First Class", "same day": "Same Day", "same-day": "Same Day"}
    df["ship_mode"] = df["ship_mode"].str.strip().str.lower().map(mode_map)
    df["region"] = df["region"].str.strip().str.title()
    return df


def apply_validity_rules(df):
    """Business rules: impossible values are treated as missing, not silently kept."""
    df = df.copy()
    m = {}
    m["neg_distance"] = int((df.distance_km <= 0).sum())
    df.loc[df.distance_km <= 0, "distance_km"] = np.nan
    m["nonpositive_weight"] = int((df.weight_kg <= 0).sum())
    df.loc[df.weight_kg <= 0, "weight_kg"] = np.nan
    bad_seq = df.delivery_date < df.ship_date
    m["delivery_before_ship"] = int(bad_seq.sum())
    df.loc[bad_seq, "delivery_date"] = pd.NaT
    METRICS["validity"] = m
    return df


def impute_missing(df):
    df = df.copy()
    METRICS["missing_before_imputation"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    # missing-indicator flags preserve the information that a value was filled in
    for c in ["weight_kg", "distance_km", "order_value_inr"]:
        df[f"{c}_was_missing"] = df[c].isna().astype(int)
    df["region"] = df["region"].fillna(df["region"].mode()[0])
    df["weight_kg"] = df["weight_kg"].fillna(df.groupby("ship_mode")["weight_kg"].transform("median"))
    df["distance_km"] = df["distance_km"].fillna(df.groupby("region")["distance_km"].transform("median"))
    # order value is strongly tied to weight -> predict it from weight via a median ratio
    ratio = (df.order_value_inr / df.weight_kg).median()
    df["order_value_inr"] = df["order_value_inr"].fillna(df.weight_kg * ratio)
    df["stock_on_hand"] = df["stock_on_hand"].fillna(df.groupby("region")["stock_on_hand"].transform("median"))
    return df


def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def handle_outliers(df, cols):
    """Detect on the log scale (skewed data), then cap (winsorise) at the IQR fences."""
    df = df.copy()
    out = {}
    for c in cols:
        lg = np.log1p(df[c])
        lo, hi = iqr_bounds(lg)
        z = (lg - lg.mean()) / lg.std()
        out[c] = {"iqr_flagged": int(((lg < lo) | (lg > hi)).sum()),
                  "zscore_flagged": int((z.abs() > 3).sum()),
                  "raw_max_before": round(float(df[c].max()), 1)}
        df[f"{c}_outlier"] = ((lg < lo) | (lg > hi)).astype(int)
        df[c] = np.expm1(lg.clip(lo, hi))
        out[c]["raw_max_after"] = round(float(df[c].max()), 1)
    METRICS["outliers"] = out
    return df


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING, ENCODING, SCALING
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df["processing_days"] = (df.ship_date - df.order_date).dt.days
    df["transit_days"] = (df.delivery_date - df.ship_date).dt.days
    df["delay_days"] = df.transit_days - df.planned_days
    df["is_late"] = (df.delay_days > 0).astype(float).where(df.transit_days.notna())
    df["order_month"] = df.order_date.dt.month
    df["order_dow"] = df.order_date.dt.dayofweek
    df["value_per_kg"] = df.order_value_inr / df.weight_kg
    return df


def encode_and_scale(df):
    mode_order = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 3}
    d = df.copy()
    d["ship_mode_ord"] = d.ship_mode.map(mode_order)
    d = pd.concat([d, pd.get_dummies(d.region, prefix="region", dtype=int)], axis=1)

    for c in ["weight_kg", "distance_km", "order_value_inr"]:          # tame right skew
        d[f"log_{c}"] = np.log1p(d[c])
    num = ["log_weight_kg", "log_distance_km", "log_order_value_inr", "stock_on_hand", "planned_days"]

    # time-based split: scaler is FITTED ON TRAIN ONLY to avoid data leakage
    cutoff = d.order_date.quantile(0.8)
    train, test = d[d.order_date <= cutoff].copy(), d[d.order_date > cutoff].copy()
    sc = StandardScaler().fit(train[num])
    train[[f"{c}_z" for c in num]] = sc.transform(train[num])
    test[[f"{c}_z" for c in num]] = sc.transform(test[num])
    METRICS["split"] = {"train_rows": len(train), "test_rows": len(test)}
    METRICS["train_z_mean_max_abs"] = float(np.abs(train[[f"{c}_z" for c in num]].mean()).max())
    METRICS["train_z_std_mean"] = float(train[[f"{c}_z" for c in num]].std().mean())
    return pd.concat([train, test]).sort_index()


# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_missing(before, after):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    cols = ["weight_kg", "distance_km", "order_value_inr", "region", "stock_on_hand", "delivery_date"]
    ax.barh(cols, [100 * before[c] for c in cols], color="#C55A11", label="Before imputation")
    ax.barh(cols, [100 * after[c] for c in cols], color="#1F4E79", label="After cleaning")
    ax.set_xlabel("% missing"); ax.invert_yaxis(); ax.legend(frameon=False); style(ax)
    fig.tight_layout(); fig.savefig("fig2_missing.png", dpi=160); plt.close(fig)


def fig_outliers(raw_w, clean_w):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].boxplot(raw_w.dropna(), vert=False, flierprops=dict(markersize=3, markerfacecolor="#C55A11", markeredgecolor="none"))
    axes[0].set_title("weight_kg before capping", fontsize=10)
    axes[1].boxplot(clean_w, vert=False, flierprops=dict(markersize=3, markerfacecolor="#1F4E79", markeredgecolor="none"))
    axes[1].set_title("weight_kg after capping", fontsize=10)
    for a in axes:
        a.set_xlabel("kg"); a.set_yticks([]); style(a)
    fig.tight_layout(); fig.savefig("fig2_outliers.png", dpi=160); plt.close(fig)


def fig_scaling(w):
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.6))
    vals = {"Raw": w.values, "Min-max": MinMaxScaler().fit_transform(w.to_frame()).ravel(),
            "Standard (z)": StandardScaler().fit_transform(w.to_frame()).ravel(),
            "Log + z": StandardScaler().fit_transform(np.log1p(w).to_frame()).ravel()}
    for ax, (k, v) in zip(axes, vals.items()):
        ax.hist(v, bins=30, color="#1F4E79"); ax.set_title(k, fontsize=9); ax.set_yticks([]); style(ax)
    fig.tight_layout(); fig.savefig("fig2_scaling.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# 6. VALIDATION
# ---------------------------------------------------------------------------
def validate(df):
    checks = {
        "no duplicate order_id": df.order_id.is_unique,
        "no missing in core columns": int(df[["weight_kg", "distance_km", "order_value_inr", "region", "ship_mode", "stock_on_hand"]].isna().sum().sum()) == 0,
        "distance > 0": bool((df.distance_km > 0).all()),
        "weight > 0": bool((df.weight_kg > 0).all()),
        "ship_date >= order_date": bool((df.ship_date >= df.order_date).all()),
        "delivery_date >= ship_date (where present)": bool((df.transit_days.dropna() >= 0).all()),
        "ship_mode has 4 valid levels": df.ship_mode.nunique() == 4,
    }
    METRICS["checks"] = {k: bool(v) for k, v in checks.items()}
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    assert all(checks.values())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = simulate_raw_shipments()
    METRICS["raw_rows"], METRICS["raw_cols"] = len(raw), raw.shape[1]
    audit(raw, "RAW")

    df = remove_duplicates(raw)
    df = fix_types(df)
    METRICS["after_dedup_rows"] = len(df)
    _t = (df.delivery_date - df.ship_date).dt.days
    METRICS["transit_mean_uncleaned"] = round(float(_t.mean()), 2)
    METRICS["transit_negative_uncleaned"] = int((_t < 0).sum())
    METRICS["late_rate_uncleaned_pct"] = round(float(100 * ((_t - df.planned_days) > 0)[_t.notna()].mean()), 1)
    METRICS["bad_dates_after_parse"] = int(df[["order_date", "ship_date"]].isna().sum().sum())
    METRICS["ship_mode_levels_raw"] = int(raw.ship_mode.nunique())
    df = standardise_categories(df)
    METRICS["ship_mode_levels_clean"] = int(df.ship_mode.nunique())
    METRICS["region_levels_raw"] = int(raw.region.nunique())
    METRICS["region_levels_clean"] = int(df.region.nunique())

    before_missing = df.isna().mean()
    df = apply_validity_rules(df)
    before_missing = df.isna().mean()
    raw_weight = df.weight_kg.copy()
    METRICS["weight_before"] = {"mean": round(float(raw_weight.mean()), 2), "median": round(float(raw_weight.median()), 2),
                                "max": round(float(raw_weight.max()), 1)}

    df = impute_missing(df)
    df = handle_outliers(df, ["weight_kg", "distance_km", "order_value_inr"])
    METRICS["weight_after"] = {"mean": round(float(df.weight_kg.mean()), 2), "median": round(float(df.weight_kg.median()), 2),
                               "max": round(float(df.weight_kg.max()), 1)}
    df = engineer_features(df)
    METRICS["late_rate_pct"] = round(float(100 * df.is_late.mean()), 1)
    METRICS["transit_mean_clean"] = round(float(df.transit_days.mean()), 2)
    METRICS["transit_missing_rows"] = int(df.transit_days.isna().sum())

    fig_missing(before_missing, df.isna().mean())
    fig_outliers(raw_weight, df.weight_kg)
    fig_scaling(df.weight_kg)

    final = encode_and_scale(df)
    validate(final)
    audit(final[["weight_kg", "distance_km", "order_value_inr", "region", "ship_mode"]], "CLEAN (core)")

    METRICS["final_rows"], METRICS["final_cols"] = len(final), final.shape[1]
    final.to_csv("clean_shipments.csv", index=False)
    json.dump(METRICS, open("week2_metrics.json", "w"), indent=2, default=str)
    print(json.dumps(METRICS, indent=2, default=str))
