"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems

Builds on the simulated shipment dataset from Week 3 (logistics_shipments_simulated.csv).

PREDICTION  : delivery delay (actual days - planned days) per shipment, known at dispatch time
              - regression models compared with time-aware validation and tuned
              - a late-delivery risk classifier used for decisions
OPTIMISATION: (A) carrier allocation by linear programming, using model-based expected cost and
              late risk; (B) dispatch-window what-if analysis using the trained model;
              (C) capacitated vehicle routing (OR-Tools) for load consolidation.

The data is SIMULATED. Results show the workflow and how model output feeds decisions; the
numbers are not claims about a real network.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", font_scale=0.85)
NAVY, ORANGE, TEAL, GREY = "#1F4E79", "#C55A11", "#2E9E8F", "#7F7F7F"
M = {}

NUM = ["distance_km", "weight_kg", "load_factor", "dwell_hours", "dispatch_hour", "planned_days", "monsoon"]
CAT = ["dest_zone", "carrier", "vehicle_type", "ship_mode"]


# ---------------------------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------------------------
def load():
    df = pd.read_csv("logistics_shipments_simulated.csv", parse_dates=["order_date"]).sort_values("order_date")
    df["monsoon"] = df.month.isin([7, 8, 9]).astype(int)
    return df.reset_index(drop=True)


def split_time(df):
    """Time-based split: train on Jan-Sep, test on the unseen Oct-Dec period."""
    train, test = df[df.order_date < "2025-10-01"], df[df.order_date >= "2025-10-01"]
    M["train_n"], M["test_n"] = len(train), len(test)
    M["target_mean_train"] = round(float(train.delay_days.mean()), 3)
    M["target_std_train"] = round(float(train.delay_days.std()), 3)
    return train, test


def preprocessor(scale=False):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


# ---------------------------------------------------------------------------
# 2. REGRESSION: MODEL COMPARISON, VALIDATION, TUNING
# ---------------------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def fit_regressors(train, test):
    Xtr, ytr, Xte, yte = train[NUM + CAT], train.delay_days, test[NUM + CAT], test.delay_days
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "Mean baseline": Pipeline([("pre", preprocessor()), ("m", DummyRegressor())]),
        "Linear regression": Pipeline([("pre", preprocessor(True)), ("m", LinearRegression())]),
        "Decision tree": Pipeline([("pre", preprocessor()), ("m", DecisionTreeRegressor(max_depth=6, min_samples_leaf=30, random_state=SEED))]),
        "Random forest": Pipeline([("pre", preprocessor()), ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=SEED))]),
        "Gradient boosting": Pipeline([("pre", preprocessor()), ("m", GradientBoostingRegressor(random_state=SEED))]),
    }

    # hyper-parameter tuning for gradient boosting with time-ordered folds (no future leakage)
    grid = {"m__n_estimators": [100, 200, 300], "m__max_depth": [2, 3, 4], "m__learning_rate": [0.03, 0.05, 0.1],
            "m__subsample": [0.7, 0.85, 1.0], "m__min_samples_leaf": [1, 10, 30, 60]}
    search = RandomizedSearchCV(models["Gradient boosting"], grid, n_iter=30, cv=tscv,
                                scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
    search.fit(Xtr, ytr)
    M["best_params"] = {k.replace("m__", ""): (float(v) if isinstance(v, float) else int(v)) for k, v in search.best_params_.items()}
    M["tuned_cv_mae"] = round(float(-search.best_score_), 3)
    models["Gradient boosting (tuned)"] = search.best_estimator_

    rows, preds = [], {}
    for name, pipe in models.items():
        cv = -cross_val_score(pipe, Xtr, ytr, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
        if name != "Gradient boosting (tuned)":
            pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte); preds[name] = p
        rows.append([name, round(float(cv.mean()), 3), round(float(cv.std()), 3), round(mean_absolute_error(yte, p), 3),
                     round(rmse(yte, p), 3), round(r2_score(yte, p), 3)])
        print(rows[-1])
    M["reg_table"] = rows
    # final model is chosen on cross-validation error (training period only), never on the test set
    cand = [r for r in rows if r[0] != "Mean baseline"]
    best = min(cand, key=lambda r: r[1])[0]
    M["best_model"] = best
    return models, preds, best


def regression_figures(models, preds, best, test):
    yte = test.delay_days.values
    # Figure 1: model comparison
    t = pd.DataFrame(M["reg_table"], columns=["model", "cv_mae", "cv_sd", "mae", "rmse", "r2"])
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    x = np.arange(len(t)); w = 0.38
    ax.bar(x - w / 2, t.cv_mae, w, color=NAVY, label="Cross-validation MAE (train)")
    ax.bar(x + w / 2, t.mae, w, color=ORANGE, label="Test MAE (Oct–Dec)")
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(").replace(" regression", "\nregression") for n in t.model], fontsize=7.5)
    ax.set_ylabel("MAE (days)"); ax.set_ylim(0, 0.8); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Model comparison: lower is better", loc="left", fontweight="bold"); sns.despine()
    fig.tight_layout(); fig.savefig("fig4_1_models.png", dpi=160); plt.close(fig)

    # Figure 2: predicted vs actual (binned) and residuals
    p = preds[best]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    bins = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"pred": p, "act": yte}).groupby(bins, observed=True).mean()
    axes[0].plot([g.pred.min(), g.pred.max()], [g.pred.min(), g.pred.max()], ls="--", color=GREY, label="Perfect")
    axes[0].plot(g.pred, g.act, marker="o", color=NAVY, label="Test decile averages")
    axes[0].set_xlabel("Predicted delay (days)"); axes[0].set_ylabel("Actual mean delay (days)"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Calibration by prediction decile", loc="left", fontweight="bold", fontsize=9)
    sns.histplot(yte - p, bins=30, color=ORANGE, ax=axes[1])
    axes[1].set_xlabel("Residual (actual − predicted, days)"); axes[1].set_ylabel("")
    axes[1].set_title("Residual distribution", loc="left", fontweight="bold", fontsize=9)
    sns.despine()
    fig.tight_layout(); fig.savefig("fig4_2_calibration.png", dpi=160); plt.close(fig)
    M["resid_mean"] = round(float((yte - p).mean()), 3)
    M["resid_std"] = round(float((yte - p).std()), 3)
    M["decile_range"] = [round(float(g.pred.min()), 2), round(float(g.pred.max()), 2), round(float(g.act.min()), 2), round(float(g.act.max()), 2)]

    # Figure 3: permutation importance on held-out data
    pi = permutation_importance(models[best], test[NUM + CAT], test.delay_days, n_repeats=10, random_state=SEED,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(imp.index, imp.values, color=[ORANGE if v == imp.max() else NAVY for v in imp.values])
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (days)")
    ax.set_title("Permutation importance of features", loc="left", fontweight="bold"); sns.despine()
    fig.tight_layout(); fig.savefig("fig4_3_importance.png", dpi=160); plt.close(fig)
    M["importance"] = [[k, round(float(v), 4)] for k, v in imp.sort_values(ascending=False).items()]


# ---------------------------------------------------------------------------
# 3. CLASSIFICATION: LATE-DELIVERY RISK
# ---------------------------------------------------------------------------
def fit_classifier(train, test):
    Xtr, ytr, Xte, yte = train[NUM + CAT], train.is_late, test[NUM + CAT], test.is_late
    lr = Pipeline([("pre", preprocessor(True)), ("m", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    gb = Pipeline([("pre", preprocessor()), ("m", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                                              subsample=0.85, random_state=SEED))]).fit(Xtr, ytr)
    p_lr, p_gb = lr.predict_proba(Xte)[:, 1], gb.predict_proba(Xte)[:, 1]
    M["auc_lr"], M["auc_gb"] = round(float(roc_auc_score(yte, p_lr)), 3), round(float(roc_auc_score(yte, p_gb)), 3)
    M["late_rate_test"] = round(float(100 * yte.mean()), 1)

    thr = 0.5 * 0 + float(np.quantile(p_gb, 0.80))           # flag the riskiest 20 % of shipments
    flag = p_gb >= thr
    M["flag_threshold"] = round(thr, 3)
    M["flag_precision"] = round(float(100 * yte[flag].mean()), 1)
    M["flag_recall"] = round(float(100 * yte[flag].sum() / yte.sum()), 1)
    M["flag_lift"] = round(float(yte[flag].mean() / yte.mean()), 2)

    fpr, tpr, _ = roc_curve(yte, p_gb); fpr2, tpr2, _ = roc_curve(yte, p_lr)
    dec = pd.DataFrame({"p": p_gb, "y": yte.values})
    dec["decile"] = pd.qcut(dec.p.rank(method="first"), 10, labels=False) + 1
    lift = dec.groupby("decile").y.mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(fpr, tpr, color=NAVY, lw=2, label=f"Gradient boosting (AUC {M['auc_gb']})")
    axes[0].plot(fpr2, tpr2, color=ORANGE, lw=1.6, label=f"Logistic regression (AUC {M['auc_lr']})")
    axes[0].plot([0, 1], [0, 1], ls="--", color=GREY)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[0].set_title("ROC curve", loc="left", fontweight="bold", fontsize=9)
    axes[1].bar(lift.index, lift.values, color=[ORANGE if d >= 9 else NAVY for d in lift.index])
    axes[1].axhline(100 * yte.mean(), ls="--", color=GREY)
    axes[1].set_xlabel("Risk decile (10 = highest predicted risk)"); axes[1].set_ylabel("Actual late rate (%)")
    axes[1].set_title("Late rate by predicted-risk decile", loc="left", fontweight="bold", fontsize=9)
    sns.despine()
    fig.tight_layout(); fig.savefig("fig4_4_classifier.png", dpi=160); plt.close(fig)
    M["decile_low"], M["decile_high"] = round(float(lift.iloc[0]), 1), round(float(lift.iloc[-1]), 1)
    return gb


# ---------------------------------------------------------------------------
# 4A. OPTIMISATION - CARRIER ALLOCATION (linear programme)
# ---------------------------------------------------------------------------
def carrier_allocation(df, test, clf, penalty=2000.0, max_share=0.55, min_share=0.15):
    carriers, zones = ["Carrier A", "Carrier B", "Carrier C"], ["North", "Central", "West", "East", "South"]
    # carrier cost effect from a log-cost regression (same idea as Week 3)
    X = pd.get_dummies(df[["carrier", "vehicle_type", "ship_mode"]], drop_first=True).astype(float)
    X["log_d"], X["log_w"] = np.log(df.distance_km), np.log(df.weight_kg)
    lr = LinearRegression().fit(X, np.log(df.cost_inr))
    eff = {"Carrier A": 0.0, "Carrier B": float(lr.coef_[list(X.columns).index("carrier_Carrier B")]),
           "Carrier C": float(lr.coef_[list(X.columns).index("carrier_Carrier C")])}
    M["carrier_cost_effect_pct"] = {c: round(float(100 * (np.exp(v) - 1)), 1) for c, v in eff.items()}

    cost = np.zeros((len(zones), len(carriers))); plate = np.zeros_like(cost)
    for i, z in enumerate(zones):
        sub = test[test.dest_zone == z]
        cur = sub.carrier.map(eff).values
        for j, c in enumerate(carriers):
            cost[i, j] = float((sub.cost_inr * np.exp(eff[c] - cur)).mean())          # expected cost if carrier c is used
            plate[i, j] = float(clf.predict_proba(sub.assign(carrier=c)[NUM + CAT])[:, 1].mean())   # expected late risk
    demand = df.dest_zone.value_counts().reindex(zones).values.astype(float)
    total = demand.sum()
    unit = cost + penalty * plate

    # baseline: today's carrier mix inside each zone
    mix = df.pivot_table(index="dest_zone", columns="carrier", values="shipment_id", aggfunc="count").reindex(zones)[carriers]
    base = (mix.div(mix.sum(axis=1), axis=0).values) * demand[:, None]

    # variables x[z, c]; minimise expected cost + penalty * expected late shipments
    nz, nc = len(zones), len(carriers)
    c_vec = unit.ravel()
    A_eq = np.zeros((nz, nz * nc)); b_eq = demand
    for i in range(nz):
        A_eq[i, i * nc:(i + 1) * nc] = 1
    A_ub, b_ub = [], []
    for j in range(nc):
        row = np.zeros(nz * nc); row[j::nc] = 1
        A_ub.append(row); b_ub.append(max_share * total)           # no carrier above max_share
        A_ub.append(-row); b_ub.append(-min_share * total)         # keep every carrier above min_share
    res = linprog(c_vec, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    assert res.success
    opt = res.x.reshape(nz, nc)

    def summarise(x):
        return {"cost": float((x * cost).sum()), "late": float((x * plate).sum()), "obj": float((x * unit).sum())}
    b, o = summarise(base), summarise(opt)
    M["alloc"] = {"penalty": penalty, "max_share": max_share, "min_share": min_share, "total": int(total),
                  "base_cost_m": round(b["cost"] / 1e6, 2), "opt_cost_m": round(o["cost"] / 1e6, 2),
                  "base_late": round(b["late"]), "opt_late": round(o["late"]),
                  "base_late_pct": round(100 * b["late"] / total, 1), "opt_late_pct": round(100 * o["late"] / total, 1),
                  "obj_saving_pct": round(100 * (b["obj"] - o["obj"]) / b["obj"], 1),
                  "cost_change_pct": round(100 * (o["cost"] - b["cost"]) / b["cost"], 1),
                  "late_change_pct": round(100 * (o["late"] - b["late"]) / b["late"], 1)}
    M["alloc_matrix"] = {z: {c: round(float(100 * opt[i, j] / demand[i]), 0) for j, c in enumerate(carriers)} for i, z in enumerate(zones)}
    M["alloc_base"] = {z: {c: round(float(100 * base[i, j] / demand[i]), 0) for j, c in enumerate(carriers)} for i, z in enumerate(zones)}
    M["plate_east"] = {c: round(float(100 * plate[zones.index("East"), j]), 1) for j, c in enumerate(carriers)}
    M["carrier_share_opt"] = {c: round(float(100 * opt[:, j].sum() / total), 1) for j, c in enumerate(carriers)}
    M["carrier_share_base"] = {c: round(float(100 * base[:, j].sum() / total), 1) for j, c in enumerate(carriers)}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    cols = [NAVY, TEAL, ORANGE]
    for ax, data, title in [(axes[0], base, "Current allocation"), (axes[1], opt, "Optimised allocation")]:
        share = data / data.sum(axis=1, keepdims=True) * 100
        left = np.zeros(nz)
        for j, c in enumerate(carriers):
            ax.barh(zones, share[:, j], left=left, color=cols[j], label=c); left += share[:, j]
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9); ax.set_xlabel("% of zone volume")
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3)
    sns.despine()
    fig.tight_layout(); fig.savefig("fig4_5_allocation.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# 4B. OPTIMISATION - DISPATCH WINDOW WHAT-IF (uses the trained model)
# ---------------------------------------------------------------------------
def dispatch_whatif(df, test, clf):
    ev = lambda d: d.dispatch_hour.between(15, 19)
    ratio = df[~ev(df)].dwell_hours.mean() / df[ev(df)].dwell_hours.mean()
    cf = test.copy()
    cf.loc[ev(cf), "dwell_hours"] = cf.loc[ev(cf), "dwell_hours"] * ratio          # staggered, better-staffed dispatch
    p0, p1 = clf.predict_proba(test[NUM + CAT])[:, 1], clf.predict_proba(cf[NUM + CAT])[:, 1]
    mask = ev(test).values
    M["whatif"] = {"dwell_ratio": round(float(ratio), 2), "evening_share": round(float(100 * mask.mean()), 1),
                   "late_before_evening": round(float(100 * p0[mask].mean()), 1), "late_after_evening": round(float(100 * p1[mask].mean()), 1),
                   "late_before_all": round(float(100 * p0.mean()), 1), "late_after_all": round(float(100 * p1.mean()), 1),
                   "avoided_per_1000": round(float(1000 * (p0.mean() - p1.mean())), 1)}
    blocks = pd.cut(test.dispatch_hour, [7, 11, 15, 19, 21], labels=["08–11", "12–15", "16–19", "20–21"])
    g = pd.DataFrame({"b": blocks.values, "before": p0 * 100, "after": p1 * 100}).groupby("b", observed=True).mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, g.before, w, color=ORANGE, label="Current dwell times")
    ax.bar(x + w / 2, g.after, w, color=NAVY, label="Dwell cut in the 15:00–19:59 window")
    ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_xlabel("Dispatch hours"); ax.set_ylabel("Predicted late risk (%)")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Model-based what-if: dispatch queue", loc="left", fontweight="bold"); sns.despine()
    fig.tight_layout(); fig.savefig("fig4_6_whatif.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# 4C. OPTIMISATION - CONSOLIDATION ROUTING (capacitated VRP)
# ---------------------------------------------------------------------------
def routing(df, n=24, capacity=4000, fixed=2200.0, perkm=26.0, perkg=0.9):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    sub = df[(df.dest_zone == "North") & (df.month == 10) & (df.weight_kg <= capacity)].sample(n, random_state=SEED)
    ang = rng.uniform(0, 2 * np.pi, n)                                   # simulated bearings; radial distance from the data
    pts = np.vstack([[0, 0], np.c_[sub.distance_km * np.cos(ang), sub.distance_km * np.sin(ang)]])
    dist = (np.linalg.norm(pts[:, None] - pts[None], axis=2) * 100).astype(int)       # 0.01 km units
    dem = [0] + sub.weight_kg.round().astype(int).tolist()
    mgr = pywrapcp.RoutingIndexManager(n + 1, n, 0)
    r = pywrapcp.RoutingModel(mgr)
    t = r.RegisterTransitCallback(lambda i, j: dist[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    r.SetArcCostEvaluatorOfAllVehicles(t)
    r.SetFixedCostOfAllVehicles(int(fixed / perkm * 100))                # fixed vehicle charge expressed in distance units
    d = r.RegisterUnaryTransitCallback(lambda i: dem[mgr.IndexToNode(i)])
    r.AddDimensionWithVehicleCapacity(d, 0, [capacity] * n, True, "Load")
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = 5
    sol = r.SolveWithParameters(p)
    routes = []
    for v in range(n):
        i, route = r.Start(v), [0]
        while not r.IsEnd(i):
            i = sol.Value(r.NextVar(i)); route.append(mgr.IndexToNode(i))
        if len(route) > 2:
            routes.append(route)
    km = lambda rt: sum(dist[a][b] for a, b in zip(rt[:-1], rt[1:])) / 100
    opt_km = sum(km(rt) for rt in routes)
    naive_km = float(sum(2 * dist[0][k] / 100 for k in range(1, n + 1)))
    kg = float(sum(dem))
    base_cost = n * fixed + perkm * naive_km + perkg * kg
    opt_cost = len(routes) * fixed + perkm * opt_km + perkg * kg
    loads = [sum(dem[k] for k in rt) / capacity for rt in routes]
    M["vrp"] = {"deliveries": n, "capacity": capacity, "routes": len(routes), "naive_km": round(naive_km), "opt_km": round(opt_km),
                "km_saving_pct": round(100 * (1 - opt_km / naive_km), 1), "base_cost": round(base_cost), "opt_cost": round(opt_cost),
                "cost_saving_pct": round(100 * (1 - opt_cost / base_cost), 1), "avg_load_pct": round(100 * float(np.mean(loads))),
                "base_load_pct": round(100 * float(np.mean([w / capacity for w in dem[1:]]))), "total_kg": round(kg)}
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    colors = sns.color_palette("tab10", len(routes))
    for rt, c in zip(routes, colors):
        ax.plot(pts[rt, 0], pts[rt, 1], marker="o", ms=4, lw=1.6, color=c)
    ax.scatter([0], [0], marker="*", s=220, color="black", zorder=5, label="Warehouse")
    ax.set_xlabel("km east"); ax.set_ylabel("km north"); ax.set_aspect("equal")
    ax.annotate("Warehouse", (0, 0), xytext=(10, 12), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.set_title(f"{len(routes)} consolidated routes serve {n} deliveries", loc="left", fontweight="bold"); sns.despine()
    fig.tight_layout(); fig.savefig("fig4_7_routes.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load()
    train, test = split_time(df)
    models, preds, best = fit_regressors(train, test)
    regression_figures(models, preds, best, test)
    clf = fit_classifier(train, test)
    carrier_allocation(df, test, clf)
    dispatch_whatif(df, test, clf)
    routing(df)
    json.dump(M, open("week4_metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in M.items() if k not in ("reg_table",)}, indent=1))
