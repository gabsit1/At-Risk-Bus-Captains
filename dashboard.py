"""
Dashboard for the CEG2003 at-risk bus captain project.

First run the pipeline:   python atrisk_pipeline.py
Then start the dashboard: streamlit run dashboard.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = Path("outputs")
st.set_page_config(page_title="At-risk Bus Captain Dashboard", layout="wide")

TYPE_LABELS = {
    "rate_harsh_acceleration": "Harsh Acceleration", "rate_harsh_braking": "Harsh Braking",
    "rate_speeding": "Speeding", "rate_swaying": "Swaying", "rate_sharp_cornering": "Sharp Cornering",
    "rate_fatigue_alert": "Fatigue Alert", "rate_yawn": "Yawn",
    "rate_headway_monitoring_alert": "Headway Alert",
    "rate_forward_collision_warning": "Fwd Collision Warning",
    "rate_pedestrian_collision_alert": "Pedestrian Alert",
}
TIER_COLORS = {"High": "#c0392b", "Medium": "#e67e22", "Low": "#27ae60", "Insufficient data": "#95a5a6"}


@st.cache_data
def load():
    drivers = pd.read_csv(OUT / "driver_risk_scores.csv")
    alarms = pd.read_csv(OUT / "alarm_summary.csv", parse_dates=["date"])
    acc = pd.read_csv(OUT / "accidents_clean.csv", parse_dates=["acc_date"])
    metrics = pd.read_csv(OUT / "model_metrics.csv") if (OUT / "model_metrics.csv").exists() else None
    imp = pd.read_csv(OUT / "feature_importance.csv") if (OUT / "feature_importance.csv").exists() else None
    return drivers, alarms, acc, metrics, imp


try:
    drivers, alarms, acc, metrics, imp = load()
except FileNotFoundError:
    st.error("Run `python atrisk_pipeline.py` first - no files found in ./outputs")
    st.stop()

# Some alarms/drivers have no operation line (NaN). Fill BEFORE converting to str,
# otherwise newer pandas keeps NaN as a float and sorted() fails on mixed types.
drivers["main_line"] = drivers["main_line"].fillna("Unknown").astype(str)
alarms["line"] = alarms["line"].fillna("Unknown").astype(str)

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Filters")
tiers = st.sidebar.multiselect("Risk tier", list(TIER_COLORS), default=["High", "Medium", "Low"])
min_days = st.sidebar.slider("Min. active days", 1, 31, 5)
lines = sorted(drivers["main_line"].unique())
sel_lines = st.sidebar.multiselect("Main operating line", lines)

f = drivers[drivers["risk_tier"].isin(tiers) & (drivers["active_days"] >= min_days)]
if sel_lines:
    f = f[f["main_line"].isin(sel_lines)]

st.title("At-risk bus captain dashboard")
st.caption("March 2024 alarm data + 2024 accident records. Confidential - do not share outside the team.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "At-risk drivers", "Driver profile", "Lines", "Model"])

# ------------------------------------------------------------------ overview
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drivers", f"{len(f):,}")
    c2.metric("Alarms (all drivers)", f"{int(alarms['n'].sum()):,}")
    c3.metric("High-risk drivers", int((f["risk_tier"] == "High").sum()))
    c4.metric("Drivers with accident", f"{f['had_accident'].mean():.1%}")

    left, right = st.columns(2)
    by_type = alarms.groupby("alarm_type")["n"].sum().sort_values().reset_index()
    left.plotly_chart(px.bar(by_type, x="n", y="alarm_type", orientation="h",
                             title="Alarms by type", labels={"n": "Alarms", "alarm_type": ""}),
                      use_container_width=True)
    by_hour = alarms.groupby(["hour", "alarm_type"])["n"].sum().reset_index()
    right.plotly_chart(px.area(by_hour, x="hour", y="n", color="alarm_type", title="Alarms by hour of day",
                               labels={"n": "Alarms", "hour": "Hour"}), use_container_width=True)

    daily = alarms.groupby("date")["n"].sum().reset_index()
    st.plotly_chart(px.line(daily, x="date", y="n", title="Daily alarm volume", markers=True,
                            labels={"n": "Alarms", "date": ""}), use_container_width=True)

    acc_m = acc.assign(month=acc["acc_date"].dt.to_period("M").astype(str)).groupby("month").size().reset_index(name="accidents")
    st.plotly_chart(px.bar(acc_m, x="month", y="accidents", title="Valid accident reports per month"),
                    use_container_width=True)

# ------------------------------------------------------------------ at-risk drivers
with tab2:
    top_n = st.slider("Show top N drivers", 10, 200, 50)
    cols = ["Driver", "risk_tier", "combined_score", "risk_score", "ml_risk_prob", "alarms_per_hour",
            "active_days", "main_line", "n_accidents",
            "rate_forward_collision_warning", "rate_harsh_braking", "rate_fatigue_alert", "rate_pedestrian_collision_alert"]
    cols = [c for c in cols if c in f.columns]
    top = f.sort_values("combined_score", ascending=False).head(top_n)[cols]
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.download_button("Download list (CSV)", top.to_csv(index=False), "at_risk_drivers.csv")

    if "ml_risk_prob" in f:
        st.plotly_chart(px.scatter(f, x="risk_score", y="ml_risk_prob", color="risk_tier",
                                   color_discrete_map=TIER_COLORS, hover_data=["Driver", "alarms_per_hour"],
                                   title="Rule-based score vs ML probability (out-of-fold)"),
                        use_container_width=True)

    # does the score actually line up with accidents?
    tier_acc = drivers.groupby("risk_tier")["had_accident"].agg(["mean", "sum", "count"]).reset_index()
    tier_acc["mean"] *= 100
    st.plotly_chart(px.bar(tier_acc, x="risk_tier", y="mean", color="risk_tier", color_discrete_map=TIER_COLORS,
                           text=tier_acc["mean"].round(1), title="% of drivers with an accident, by risk tier",
                           labels={"mean": "% with accident", "risk_tier": ""}), use_container_width=True)

# ------------------------------------------------------------------ driver profile
with tab3:
    did = st.selectbox("Driver ID", sorted(drivers["Driver"].unique()))
    d = drivers[drivers["Driver"] == did].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk tier", d["risk_tier"])
    m2.metric("Combined score", f"{d['combined_score']:.0f}")
    m3.metric("Alarms / active hour", f"{d['alarms_per_hour']:.2f}",
              delta=f"{d['alarms_per_hour'] - drivers['alarms_per_hour'].median():+.2f} vs median")
    m4.metric("Accidents on record", int(d["n_accidents"]))

    pct = {TYPE_LABELS[c]: (drivers[c] < d[c]).mean() * 100 for c in TYPE_LABELS}
    fig = go.Figure(go.Scatterpolar(r=list(pct.values()) + [list(pct.values())[0]],
                                    theta=list(pct.keys()) + [list(pct.keys())[0]], fill="toself",
                                    name=f"Driver {did}", line_color="#c0392b"))
    fig.add_trace(go.Scatterpolar(r=[50] * (len(pct) + 1), theta=list(pct.keys()) + [list(pct.keys())[0]],
                                  name="Fleet median", line=dict(dash="dash", color="#7f8c8d")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                      title="Alarm rate percentile vs all drivers (100 = worst)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Main line {d['main_line']} | {int(d['active_days'])} active days | "
               f"night share {d['night_share']:.1%} | high-speed alarm share {d['highspeed_share']:.1%}")
    ad = acc[acc["Driver"] == did]
    if len(ad):
        st.write("Accident records")
        st.dataframe(ad, hide_index=True)

# ------------------------------------------------------------------ lines
with tab4:
    ld = alarms.groupby("line")["n"].sum().rename("alarms").reset_index()
    nd = drivers.groupby("main_line").agg(drivers=("Driver", "count"), mean_risk=("combined_score", "mean"),
                                          acc_rate=("had_accident", "mean")).reset_index().rename(columns={"main_line": "line"})
    lt = ld.merge(nd, on="line", how="left")
    lt["alarms_per_driver"] = lt["alarms"] / lt["drivers"]
    lt = lt[lt["drivers"] >= 5].sort_values("alarms_per_driver", ascending=False)
    st.plotly_chart(px.bar(lt.head(25), x="line", y="alarms_per_driver",
                           title="Lines with most alarms per driver (>=5 drivers)"), use_container_width=True)
    st.dataframe(lt, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ model
with tab5:
    if metrics is not None:
        st.subheader("Cross-validated performance")
        st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
        st.caption("ROC-AUC 0.5 = random. PR-AUC should be compared with the positive rate. "
                   "Baseline rows are simple scores with no training.")
    if imp is not None:
        st.plotly_chart(px.bar(imp.head(15).iloc[::-1], x="rf_permutation_importance", y="feature",
                               orientation="h", title="Feature importance (Random Forest)"),
                        use_container_width=True)