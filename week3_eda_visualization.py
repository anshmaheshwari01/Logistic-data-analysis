"""
Week 3 - Advanced Data Analysis and Visualization in Logistics

Simulates 8,000 outbound shipments from a Delhi NCR warehouse during 2025, runs an
exploratory data analysis, produces nine figures (matplotlib + seaborn) and saves the
key numbers to week3_metrics.json for the written report.

IMPORTANT: the data is SIMULATED. Realistic relationships (distance and vehicle size drive
cost, evening dispatch batches create dwell time, monsoon and festive peaks add delay, one
carrier is weaker on one zone) are built in so that the analysis has structure to discover.
Findings therefore demonstrate the method; they are not claims about a real company.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(11)
sns.set_theme(style="whitegrid", font_scale=0.85)
NAVY, ORANGE, GREY = "#1F4E79", "#C55A11", "#7F7F7F"
PAL = [NAVY, ORANGE, "#2E9E8F", "#8E6BBF"]
M = {}


# ---------------------------------------------------------------------------
# 1. DATA SIMULATION
# ---------------------------------------------------------------------------
def simulate(n=8000):
    days = pd.date_range("2025-01-01", "2025-12-31")
    w = np.ones(len(days))
    w[days.month.isin([10, 11])] *= 1.5                 # festive-season peak
    w[days.month == 12] *= 1.2
    w[days.dayofweek == 6] *= 0.55                      # quieter Sundays
    date = rng.choice(days, n, p=w / w.sum())

    zone = rng.choice(["North", "Central", "West", "East", "South"], n, p=[0.30, 0.12, 0.22, 0.14, 0.22])
    log_mu = {"North": 5.1, "Central": 6.4, "West": 6.9, "East": 7.15, "South": 7.4}
    sig = {"North": 0.55, "Central": 0.2, "West": 0.2, "East": 0.18, "South": 0.15}
    distance = np.clip([rng.lognormal(log_mu[z], sig[z]) for z in zone], 8, 2200)
    weight = np.clip(rng.lognormal(6.6, 0.9, n), 20, 9000)

    # vehicle choice: mostly matched to weight, sometimes an oversized vehicle is used
    cap = {"Mini Van": 1000, "Truck 4T": 4000, "Truck 10T": 10000}
    veh = []
    for wt, d in zip(weight, distance):
        if wt <= 1000:
            v = "Mini Van" if rng.random() < (0.88 if d < 300 else 0.55) else "Truck 4T"
        elif wt <= 4000:
            v = "Truck 4T" if rng.random() < 0.88 else "Truck 10T"
        else:
            v = "Truck 10T"
        veh.append(v)
    veh = np.array(veh)
    load_factor = np.array([min(w_ / cap[v], 1.0) for w_, v in zip(weight, veh)])

    carrier = rng.choice(["Carrier A", "Carrier B", "Carrier C"], n, p=[0.45, 0.35, 0.20])
    mode = rng.choice(["Standard", "Express"], n, p=[0.70, 0.30])
    hp = np.array([2, 4, 6, 7, 8, 8, 9, 10, 10, 9, 7, 5, 3, 2], dtype=float)
    hour = rng.choice(np.arange(8, 22), n, p=hp / hp.sum())

    df = pd.DataFrame({"order_date": pd.to_datetime(date), "dispatch_hour": hour, "dest_zone": zone,
                       "carrier": carrier, "vehicle_type": veh, "ship_mode": mode,
                       "distance_km": distance.round(1), "weight_kg": weight.round(1),
                       "load_factor": load_factor})
    df["month"] = df.order_date.dt.month
    df["weekday"] = df.order_date.dt.day_name()

    # cost model: fixed + per-km (vehicle & carrier specific) + per-kg + mode surcharge + fuel drift
    fixed = df.vehicle_type.map({"Mini Van": 800, "Truck 4T": 2200, "Truck 10T": 4500})
    perkm = df.vehicle_type.map({"Mini Van": 14, "Truck 4T": 26, "Truck 10T": 42})
    cmult = df.carrier.map({"Carrier A": 1.0, "Carrier B": 0.93, "Carrier C": 1.08})
    surch = df.ship_mode.map({"Standard": 1.0, "Express": 1.25})
    fuel = 1 + 0.05 * (df.month - 1) / 11
    df["cost_inr"] = ((fixed + perkm * df.distance_km * cmult + 0.9 * df.weight_kg) * surch * fuel
                      * rng.lognormal(0, 0.08, n)).round(0)
    df["cost_per_kg"] = df.cost_inr / df.weight_kg

    # warehouse dwell: evening batch dispatch and busy days create queues
    daily = df.groupby("order_date").order_date.transform("count")
    busy = 1 + 0.5 * (daily / daily.mean() - 1)
    evening = np.where(df.dispatch_hour.between(15, 19), 2.3, 1.0)
    df["dwell_hours"] = (rng.gamma(2.5, 2.4, n) * evening * busy).round(1)

    # transit: planned vs actual days
    plan_std = 1 + np.ceil(df.distance_km / 450)
    plan_exp = np.maximum(1, np.ceil(df.distance_km / 700))
    df["planned_days"] = np.where(df.ship_mode == "Standard", plan_std, plan_exp).astype(int)
    monsoon = df.month.isin([7, 8, 9]).astype(int)
    c_east = ((df.carrier == "Carrier C") & (df.dest_zone == "East")).astype(int)
    raw = (rng.normal(-0.75, 0.65, n) + 0.035 * df.dwell_hours + 0.0004 * df.distance_km
           + 0.55 * monsoon + 0.9 * c_east)
    df["actual_days"] = np.maximum(1, np.round(df.planned_days + raw)).astype(int)
    df["delay_days"] = df.actual_days - df.planned_days
    df["is_late"] = (df.delay_days > 0).astype(int)
    df["time_block"] = pd.cut(df.dispatch_hour, [7, 11, 15, 19, 21],
                              labels=["08–11", "12–15", "16–19", "20–21"])
    df["shipment_id"] = np.arange(500000, 500000 + n)
    return df[["shipment_id", "order_date", "month", "weekday", "dispatch_hour", "time_block", "dest_zone",
               "carrier", "vehicle_type", "ship_mode", "distance_km", "weight_kg", "load_factor",
               "cost_inr", "cost_per_kg", "dwell_hours", "planned_days", "actual_days", "delay_days", "is_late"]]


# ---------------------------------------------------------------------------
# 2. EXPLORATORY STATISTICS
# ---------------------------------------------------------------------------
def describe(df):
    cols = ["distance_km", "weight_kg", "load_factor", "cost_inr", "cost_per_kg", "dwell_hours",
            "planned_days", "actual_days", "delay_days"]
    d = df[cols].agg(["mean", "median", "std", "skew"]).T
    d["p95"] = df[cols].quantile(0.95)
    M["describe"] = [[c] + [round(float(x), 2) for x in d.loc[c]] for c in d.index]
    M["n"], M["cols"] = len(df), df.shape[1]
    M["date_min"], M["date_max"] = str(df.order_date.min().date()), str(df.order_date.max().date())
    M["on_time_pct"] = round(float(100 * (1 - df.is_late.mean())), 1)
    M["mean_delay_late"] = round(float(df.loc[df.is_late == 1, "delay_days"].mean()), 2)
    print(d.round(2))


def correlations(df):
    cols = ["distance_km", "weight_kg", "load_factor", "cost_inr", "cost_per_kg", "dwell_hours",
            "planned_days", "delay_days"]
    corr = df[cols].corr(method="spearman")
    pairs = {"dist_cost": ("distance_km", "cost_inr"), "weight_cost": ("weight_kg", "cost_inr"),
             "load_costkg": ("load_factor", "cost_per_kg"), "dwell_delay": ("dwell_hours", "delay_days"),
             "dist_delay": ("distance_km", "delay_days"), "weight_costkg": ("weight_kg", "cost_per_kg")}
    M["corr"] = {k: round(float(corr.loc[a, b]), 2) for k, (a, b) in pairs.items()}
    return corr


# ---------------------------------------------------------------------------
# 3. VISUALISATIONS
# ---------------------------------------------------------------------------
def clean(ax):
    sns.despine(ax=ax)


def fig1_trend(df):
    g = df.groupby("month").agg(n=("shipment_id", "count"), cost=("cost_inr", "mean"))
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.bar(g.index, g.n, color=NAVY, alpha=0.85, label="Shipments")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Shipments"); ax.grid(False)
    ax2 = ax.twinx(); ax2.plot(g.index, g.cost, color=ORANGE, marker="o", lw=2, label="Avg cost (INR)")
    ax2.set_ylabel("Average cost per shipment (INR)"); ax2.grid(False)
    ax.set_title("Monthly shipment volume and average cost", loc="left", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("fig3_1_trend.png", dpi=160); plt.close(fig)
    M["peak_month"] = int(g.n.idxmax()); M["peak_month_n"] = int(g.n.max()); M["low_month_n"] = int(g.n.min())
    M["octnov_share"] = round(float(100 * g.loc[[10, 11], "n"].sum() / g.n.sum()), 1)
    M["cost_jan"], M["cost_dec"] = round(float(g.cost.iloc[0])), round(float(g.cost.iloc[-1]))


def fig2_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    sns.histplot(df.distance_km, bins=40, color=NAVY, ax=axes[0]); axes[0].set_xlabel("Distance (km)")
    sns.histplot(df.weight_kg, bins=40, color=ORANGE, log_scale=True, ax=axes[1]); axes[1].set_xlabel("Weight (kg, log scale)")
    sns.histplot(df.cost_inr, bins=40, color="#2E9E8F", log_scale=True, ax=axes[2]); axes[2].set_xlabel("Cost (INR, log scale)")
    for a in axes:
        a.set_ylabel(""); clean(a)
    axes[0].set_ylabel("Shipments")
    fig.suptitle("Distribution of distance, weight and cost", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(); fig.savefig("fig3_2_distributions.png", dpi=160); plt.close(fig)


def fig3_delay(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    vc = df.delay_days.value_counts(normalize=True).sort_index() * 100
    axes[0].bar(vc.index, vc.values, color=[NAVY if i <= 0 else ORANGE for i in vc.index])
    axes[0].set_xlabel("Delay vs plan (days)"); axes[0].set_ylabel("% of shipments")
    axes[0].set_title("Delivery delay distribution", loc="left", fontweight="bold", fontsize=9)
    order = df.groupby("dest_zone").delay_days.mean().sort_values().index
    sns.boxplot(data=df, x="dest_zone", y="delay_days", order=order, color=NAVY, fliersize=2, ax=axes[1],
                boxprops=dict(alpha=0.75), medianprops=dict(color=ORANGE, lw=2))
    axes[1].set_xlabel("Destination zone"); axes[1].set_ylabel("Delay (days)")
    axes[1].set_title("Delay by destination zone", loc="left", fontweight="bold", fontsize=9)
    for a in axes:
        clean(a)
    fig.tight_layout(); fig.savefig("fig3_3_delay.png", dpi=160); plt.close(fig)
    M["delay_mean_by_zone"] = {k: round(float(v), 2) for k, v in df.groupby("dest_zone").delay_days.mean().sort_values().items()}


def fig4_corr(corr):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlation matrix", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig("fig3_4_corr.png", dpi=160); plt.close(fig)


def fig5_scatter(df):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    sns.scatterplot(data=df.sample(3000, random_state=1), x="distance_km", y="cost_inr", hue="vehicle_type",
                    hue_order=["Mini Van", "Truck 4T", "Truck 10T"], palette=[NAVY, ORANGE, "#2E9E8F"],
                    alpha=0.45, s=14, linewidth=0, ax=ax)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Distance (km, log scale)"); ax.set_ylabel("Cost (INR, log scale)")
    ax.set_title("Cost rises with distance, in separate bands by vehicle type", loc="left", fontweight="bold")
    ax.legend(title="", frameon=False); clean(ax)
    fig.tight_layout(); fig.savefig("fig3_5_scatter.png", dpi=160); plt.close(fig)


def fig6_load(df):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["<20 %", "20–40 %", "40–60 %", "60–80 %", ">80 %"]
    df = df.assign(lf_bin=pd.cut(df.load_factor, bins, labels=labels, right=False))
    g = df.groupby("lf_bin", observed=True).agg(med=("cost_per_kg", "median"), n=("cost_per_kg", "size"))
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    bars = ax.bar(g.index.astype(str), g.med, color=[ORANGE, ORANGE, NAVY, NAVY, NAVY][:len(g)])
    for b, n in zip(bars, g.n):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}", ha="center", va="bottom", fontsize=7.5, color=GREY)
    ax.set_xlabel("Vehicle load factor"); ax.set_ylabel("Median cost per kg (INR)")
    ax.set_title("Under-filled vehicles cost more per kilogram", loc="left", fontweight="bold"); clean(ax)
    fig.tight_layout(); fig.savefig("fig3_6_loadfactor.png", dpi=160); plt.close(fig)
    M["lf_table"] = [[str(i), round(float(r.med), 1), int(r.n)] for i, r in g.iterrows()]
    M["lf_ratio"] = round(float(g.med.iloc[0] / g.med.iloc[-1]), 1)
    M["share_below_40"] = round(float(100 * (df.load_factor < 0.4).mean()), 1)


def fig7_bottleneck(df):
    g = df.groupby("dispatch_hour").agg(dwell=("dwell_hours", "mean"), late=("is_late", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    axes[0].bar(g.index, g.dwell, color=[ORANGE if 15 <= h <= 19 else NAVY for h in g.index])
    axes[0].set_xlabel("Dispatch hour"); axes[0].set_ylabel("Mean dwell time (hours)")
    axes[0].set_title("Warehouse dwell by hour", loc="left", fontweight="bold", fontsize=9)
    fest = df.month.isin([10, 11])
    for lab, mask, col in [("Oct–Nov (peak)", fest, ORANGE), ("Other months", ~fest, NAVY)]:
        gg = df[mask].groupby("dispatch_hour").is_late.mean() * 100
        axes[1].plot(gg.index, gg.values, marker="o", ms=3.5, lw=1.8, color=col, label=lab)
    axes[1].set_xlabel("Dispatch hour"); axes[1].set_ylabel("Late shipments (%)")
    axes[1].set_title("Late rate by hour", loc="left", fontweight="bold", fontsize=9); axes[1].legend(frameon=False, fontsize=8)
    for a in axes:
        clean(a)
    fig.tight_layout(); fig.savefig("fig3_7_bottleneck.png", dpi=160); plt.close(fig)
    tb = df.groupby("time_block", observed=True).agg(dwell=("dwell_hours", "mean"), late=("is_late", "mean"), n=("is_late", "size"))
    M["timeblock"] = [[str(i), round(float(r.dwell), 1), round(float(100 * r.late), 1), int(r.n)] for i, r in tb.iterrows()]
    a, b = df[df.dispatch_hour.between(15, 19)].dwell_hours, df[~df.dispatch_hour.between(15, 19)].dwell_hours
    M["dwell_evening"], M["dwell_other"] = round(float(a.mean()), 1), round(float(b.mean()), 1)
    M["evening_share"] = round(float(100 * df.dispatch_hour.between(15, 19).mean()), 1)
    M["mwu_p"] = float(stats.mannwhitneyu(a, b).pvalue)
    fm = df[fest].groupby(df.dispatch_hour.between(15, 19)).is_late.mean()
    M["late_evening_peak"], M["late_other_peak"] = round(float(100 * fm[True]), 1), round(float(100 * fm[False]), 1)


def fig8_carrier_zone(df):
    pv = df.pivot_table(index="carrier", columns="dest_zone", values="is_late", aggfunc="mean") * 100
    pv = pv[["North", "Central", "West", "East", "South"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    sns.heatmap(pv, annot=True, fmt=".0f", cmap="Oranges", linewidths=0.5, cbar_kws={"label": "% late", "shrink": 0.85}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title("Late shipments (%) by carrier and zone", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig("fig3_8_carrier_zone.png", dpi=160); plt.close(fig)
    st = pv.stack(); M["worst_cell"] = [st.idxmax()[0], st.idxmax()[1], round(float(st.max()), 1)]
    M["best_cell_val"] = round(float(st.min()), 1)
    M["carrier_late"] = {k: round(float(v), 1) for k, v in (df.groupby("carrier").is_late.mean() * 100).items()}
    cz = df[(df.carrier == "Carrier C") & (df.dest_zone == "East")]
    M["cz_n"] = int(len(cz))
    oth = df[(df.dest_zone == "East") & (df.carrier != "Carrier C")]
    M["east_other_late"] = round(float(100 * oth.is_late.mean()), 1)


def fig9_monthly_late(df):
    g = df.groupby("month").is_late.mean() * 100
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.axvspan(6.5, 9.5, color=ORANGE, alpha=0.12, label="Monsoon (Jul–Sep)")
    ax.plot(g.index, g.values, color=NAVY, marker="o", lw=2)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Late shipments (%)"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("Late-delivery rate by month", loc="left", fontweight="bold"); clean(ax)
    fig.tight_layout(); fig.savefig("fig3_9_monthly_late.png", dpi=160); plt.close(fig)
    mon = df.month.isin([7, 8, 9])
    M["late_monsoon"], M["late_nonmonsoon"] = round(float(100 * df[mon].is_late.mean()), 1), round(float(100 * df[~mon].is_late.mean()), 1)
    M["late_oct_nov"] = round(float(100 * df[df.month.isin([10, 11])].is_late.mean()), 1)


# ---------------------------------------------------------------------------
# 4. COST-DRIVER MODEL
# ---------------------------------------------------------------------------
def cost_model(df):
    X = pd.DataFrame({"log_distance": np.log(df.distance_km), "log_weight": np.log(df.weight_kg),
                      "express": (df.ship_mode == "Express").astype(int),
                      "truck_4T": (df.vehicle_type == "Truck 4T").astype(int),
                      "truck_10T": (df.vehicle_type == "Truck 10T").astype(int),
                      "carrier_B": (df.carrier == "Carrier B").astype(int),
                      "carrier_C": (df.carrier == "Carrier C").astype(int)})
    y = np.log(df.cost_inr)
    lr = LinearRegression().fit(X, y)
    rows = []
    for c, b in zip(X.columns, lr.coef_):
        rows.append([c, round(float(b), 3), round(float(b), 2) if c.startswith("log_") else round(float(100 * (np.exp(b) - 1)), 1)])
    M["reg"] = rows
    M["reg_r2"] = round(float(lr.score(X, y)), 3)
    # illustrative right-sizing: light loads (<=1000 kg) that travelled in a truck instead of a mini van
    light = (df.weight_kg <= 1000) & (df.vehicle_type != "Mini Van")
    f = np.where(df.vehicle_type == "Truck 4T", np.exp(-lr.coef_[3]), np.exp(-lr.coef_[4]))
    saving = (df.cost_inr * (1 - f))[light].sum()
    M["light_in_truck_n"] = int(light.sum())
    M["light_in_truck_pct"] = round(float(100 * light.mean()), 1)
    M["rightsize_saving_pct"] = round(float(100 * saving / df.cost_inr.sum()), 1)
    print("R2", M["reg_r2"]); print(pd.DataFrame(rows, columns=["term", "coef", "effect"]))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = simulate()
    df.to_csv("logistics_shipments_simulated.csv", index=False)
    describe(df)
    corr = correlations(df)
    fig1_trend(df); fig2_distributions(df); fig3_delay(df); fig4_corr(corr); fig5_scatter(df)
    fig6_load(df); fig7_bottleneck(df); fig8_carrier_zone(df); fig9_monthly_late(df)
    cost_model(df)
    veh = df.groupby("vehicle_type").agg(share=("shipment_id", "size"), med_lf=("load_factor", "median"),
                                         med_ckg=("cost_per_kg", "median"), med_cost=("cost_inr", "median"))
    M["vehicle"] = [[i, round(float(100 * r.share / len(df)), 1), round(float(100 * r.med_lf), 0), round(float(r.med_ckg), 1), round(float(r.med_cost))] for i, r in veh.iterrows()]
    json.dump(M, open("week3_metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in M.items() if k not in ("describe",)}, indent=1))
