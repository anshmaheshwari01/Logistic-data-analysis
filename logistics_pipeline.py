"""
Week 1 - Strategic Planning and Data Exploration in Logistics
End-to-end illustrative pipeline for a regional distributor in Delhi NCR.

Stages: data generation -> cleaning -> KPIs -> EDA -> demand forecasting (regression)
        -> SKU clustering (ABC/XYZ + KMeans) -> reorder-point optimisation
        -> route optimisation (CVRP with OR-Tools).

NOTE: Data here is SYNTHETIC so the script runs anywhere. In the real project the
`generate_data()` step is replaced by loaders for public datasets (see report, Sec. 3).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)


# ----------------------------------------------------------------------------
# 1. DATA COLLECTION (synthetic stand-in for ERP / WMS / TMS extracts)
# ----------------------------------------------------------------------------
def generate_data(n_skus=60, n_days=365):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for sku in range(n_skus):
        base = rng.gamma(shape=2.0, scale=rng.choice([5, 15, 40]))
        trend = rng.normal(0.0, 0.02)
        for i, d in enumerate(dates):
            season = 1 + 0.25 * np.sin(2 * np.pi * i / 365 + sku)
            weekly = 1.2 if d.dayofweek in (4, 5) else 1.0
            promo = int(rng.random() < 0.06)
            mean = max(base * season * weekly * (1 + trend * i / 30) * (1.5 if promo else 1), 0.1)
            rows.append((d, f"SKU{sku:03d}", rng.poisson(mean), promo,
                         round(rng.uniform(50, 900), 2), rng.integers(2, 10)))
    demand = pd.DataFrame(rows, columns=["date", "sku", "units", "promo", "unit_cost", "lead_time_days"])
    # inject realistic dirt: missing values, duplicates, negative units
    demand.loc[rng.choice(demand.index, 150, replace=False), "units"] = np.nan
    demand.loc[rng.choice(demand.index, 30, replace=False), "units"] = -1
    demand = pd.concat([demand, demand.sample(50, random_state=1)], ignore_index=True)

    n = 2000
    deliveries = pd.DataFrame({
        "order_id": np.arange(n),
        "promised_days": rng.integers(1, 4, n),
        "actual_days": rng.integers(1, 6, n),
        "qty_ordered": rng.integers(10, 100, n),
    })
    deliveries["qty_delivered"] = (deliveries.qty_ordered * rng.choice([1, 1, 1, 0.9, 0.8], n)).astype(int)
    deliveries["route_km"] = rng.uniform(8, 60, n)
    deliveries["cost_inr"] = deliveries.route_km * rng.uniform(14, 22, n)
    return demand, deliveries


# ----------------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------------
def clean_demand(df):
    df = df.drop_duplicates(["date", "sku"]).copy()
    df.loc[df["units"] < 0, "units"] = np.nan
    df = df.sort_values(["sku", "date"])
    df["units"] = df.groupby("sku")["units"].transform(lambda s: s.interpolate(limit_direction="both"))
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. KPI COMPUTATION
# ----------------------------------------------------------------------------
def compute_kpis(demand, deliveries, avg_inventory_units):
    otif = ((deliveries.actual_days <= deliveries.promised_days) &
            (deliveries.qty_delivered >= deliveries.qty_ordered)).mean()
    fill_rate = deliveries.qty_delivered.sum() / deliveries.qty_ordered.sum()
    cost_per_km = deliveries.cost_inr.sum() / deliveries.route_km.sum()
    turnover = demand.units.sum() / avg_inventory_units
    return {"OTIF_%": round(float(100 * otif), 1), "Fill_rate_%": round(float(100 * fill_rate), 1),
            "Cost_per_km_INR": round(float(cost_per_km), 2), "Inventory_turnover_x": round(float(turnover), 2)}


# ----------------------------------------------------------------------------
# 4. EDA
# ----------------------------------------------------------------------------
def run_eda(demand):
    daily = demand.groupby("date").units.sum()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(daily.index, daily.values, lw=0.8, color="#1F4E79", label="Daily units")
    ax.plot(daily.index, daily.rolling(7).mean(), lw=2, color="#C55A11", label="7-day mean")
    ax.set_title("Total daily demand (synthetic data)")
    ax.set_ylabel("Units"); ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("fig_demand.png", dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------
# 5. PREDICTIVE MODELLING - demand forecasting with lag features
# ----------------------------------------------------------------------------
def make_features(df):
    df = df.copy()
    g = df.groupby("sku").units
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = g.shift(lag)
    df["roll7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["dow"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    return df.dropna()


def forecast(demand):
    feats = make_features(demand)
    cutoff = feats.date.max() - pd.Timedelta(days=30)          # time-based split, no leakage
    train, test = feats[feats.date <= cutoff], feats[feats.date > cutoff]
    cols = ["lag_1", "lag_7", "lag_14", "roll7", "dow", "month", "promo"]

    results = {}
    naive = test["lag_7"]                                       # seasonal-naive baseline
    results["Seasonal naive"] = mean_absolute_error(test.units, naive)
    lr = LinearRegression().fit(train[cols], train.units)
    results["Linear regression"] = mean_absolute_error(test.units, lr.predict(test[cols]))
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=0)
    gb.fit(train[cols], train.units)
    pred = gb.predict(test[cols])
    results["Gradient boosting"] = mean_absolute_error(test.units, pred)
    wape = np.abs(test.units - pred).sum() / test.units.sum()
    return results, wape, test.assign(pred=pred)


# ----------------------------------------------------------------------------
# 6. CLUSTERING - ABC (value) x XYZ (variability) + KMeans segmentation
# ----------------------------------------------------------------------------
def segment_skus(demand):
    s = demand.groupby("sku").agg(mean_units=("units", "mean"), std_units=("units", "std"),
                                  unit_cost=("unit_cost", "mean"), lead_time=("lead_time_days", "mean"))
    s["cv"] = s.std_units / s.mean_units
    s["annual_value"] = s.mean_units * 365 * s.unit_cost
    s = s.sort_values("annual_value", ascending=False)
    cum = s.annual_value.cumsum() / s.annual_value.sum()
    s["ABC"] = np.where(cum <= 0.8, "A", np.where(cum <= 0.95, "B", "C"))
    X = StandardScaler().fit_transform(s[["mean_units", "cv", "unit_cost", "lead_time"]])
    s["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    return s


# ----------------------------------------------------------------------------
# 7. OPTIMISATION - safety stock & reorder point (service level 95 %)
# ----------------------------------------------------------------------------
def reorder_points(seg, z=1.645):
    seg = seg.copy()
    seg["safety_stock"] = z * seg.std_units * np.sqrt(seg.lead_time)
    seg["reorder_point"] = seg.mean_units * seg.lead_time + seg.safety_stock
    return seg


# ----------------------------------------------------------------------------
# 8. OPTIMISATION - capacitated vehicle routing (OR-Tools)
# ----------------------------------------------------------------------------
def solve_vrp(n_customers=15, n_vehicles=3, capacity=60):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    pts = rng.uniform(0, 30, size=(n_customers + 1, 2))        # index 0 = depot (km grid)
    dist = (np.linalg.norm(pts[:, None] - pts[None], axis=2) * 1000).astype(int)
    demand = [0] + list(rng.integers(5, 20, n_customers))

    mgr = pywrapcp.RoutingIndexManager(len(pts), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(mgr)
    t = routing.RegisterTransitCallback(lambda i, j: dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    routing.SetArcCostEvaluatorOfAllVehicles(t)
    d = routing.RegisterUnaryTransitCallback(lambda i: demand[mgr.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(d, 0, [capacity] * n_vehicles, True, "Load")
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = 3
    sol = routing.SolveWithParameters(p)

    total, routes = 0, []
    for v in range(n_vehicles):
        i, r = routing.Start(v), [0]
        while not routing.IsEnd(i):
            nxt = sol.Value(routing.NextVar(i))
            total += dist[mgr.IndexToNode(i)][mgr.IndexToNode(nxt)]
            r.append(mgr.IndexToNode(nxt)); i = nxt
        routes.append(r)
    # naive baseline: one customer per trip from the depot
    naive = sum(2 * dist[0][k] for k in range(1, n_customers + 1))
    return routes, total / 1000, naive / 1000


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    demand_raw, deliveries = generate_data()
    print(f"Raw rows: {len(demand_raw)}  | missing units: {demand_raw.units.isna().sum()}")
    demand = clean_demand(demand_raw)
    print(f"Clean rows: {len(demand)}  | missing units: {demand.units.isna().sum()}")

    print("KPIs:", compute_kpis(demand, deliveries, avg_inventory_units=demand.units.sum() / 12))
    run_eda(demand)

    res, wape, _ = forecast(demand)
    print("MAE by model:", {k: round(v, 2) for k, v in res.items()}, "| GB WAPE:", round(wape, 3))

    seg = reorder_points(segment_skus(demand))
    print(seg.ABC.value_counts().to_dict(), seg.cluster.value_counts().to_dict())
    print(seg[["mean_units", "cv", "safety_stock", "reorder_point"]].head(3).round(1))

    routes, opt_km, naive_km = solve_vrp()
    print(f"VRP routes: {routes}")
    print(f"Optimised {opt_km:.1f} km vs naive {naive_km:.1f} km -> {100*(1-opt_km/naive_km):.0f}% shorter")
