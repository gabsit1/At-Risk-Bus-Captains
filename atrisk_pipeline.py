"""
CEG2003 Group Project - Identifying "at-risk" bus captains
===========================================================
Pipeline:
  1. Load + clean the March 2024 alarm data (~4M rows) and the accident database
  2. Engineer one row of features per driver
  3. Build an unsupervised, explainable Risk Score (works without labels)
  4. Train / evaluate ML models that predict accident involvement
  5. Export CSVs + charts for the report and the dashboard

Run:   python atrisk_pipeline.py
Edit the CONFIG block below to point at your files.

NOTE: the data is confidential (see briefing) - keep the outputs on your own machine / team drive.
"""
import csv
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_DIR = Path("data")                       # folder holding the raw files
ALARM_FILES = ["Mar24_part1.xlsx", "Mar24_part2.xlsx"]   # .xlsx or .csv both work
ACCIDENT_FILE = "Traffic_Accident_Database_2024.xlsx"
OUT_DIR = Path("outputs")
CACHE_DIR = Path("cache")

# Which accidents count as a "positive" label?
#   "future" : accidents AFTER the alarm window (Apr-May)  -> true prediction, no time leakage
#   "march"  : accidents inside the alarm window (March)
#   "q1"     : Jan-Mar (what the briefing says the file contains)
#   "all"    : every valid accident in the file (Jan-May)
LABEL_MODE = "all"

# Accident-report flags that mean "not a real, distinct accident"
EXCLUDE_ACCIDENT_FLAGS = {"DUPLICATE", "DD", "NEAR MISS", "NM"}

MIN_ACTIVE_DAYS = 5      # drivers with fewer active days are flagged "low exposure"
TOP_PCT_AT_RISK = 0.10   # top 10% of combined score = "High risk"
RANDOM_STATE = 42

ALARM_TYPES = [
    "Harsh Acceleration", "Harsh Braking", "Speeding", "Swaying", "Sharp Cornering",
    "Fatigue Alert", "Yawn", "Headway Monitoring Alert",
    "Forward Collision Warning", "Pedestrian Collision Alert",
]
GROUPS = {
    "behaviour": ["Harsh Acceleration", "Harsh Braking", "Speeding", "Swaying", "Sharp Cornering"],
    "fatigue": ["Fatigue Alert", "Yawn"],
    "collision": ["Headway Monitoring Alert", "Forward Collision Warning", "Pedestrian Collision Alert"],
}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ----------------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------------
def xlsx_to_csv(xlsx_path, csv_path):
    """Stream a huge .xlsx to CSV row-by-row (pandas.read_excel can run out of RAM on 100MB files)."""
    from openpyxl import load_workbook
    print(f"  converting {xlsx_path.name} -> csv (one-off, several minutes)...")
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        wrote_header = False
        for ws in wb.worksheets:
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i == 0:
                    if not wrote_header:
                        w.writerow(row)
                        wrote_header = True
                    continue
                w.writerow(row)
    wb.close()


def load_alarms():
    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    for name in ALARM_FILES:
        p = DATA_DIR / name
        if p.suffix.lower() == ".csv":
            csv_path = p
        else:
            csv_path = CACHE_DIR / (p.stem + ".csv")
            if not csv_path.exists():
                xlsx_to_csv(p, csv_path)
        frames.append(pd.read_csv(csv_path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    return df


def clean_alarms(df):
    n0 = len(df)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"speed(km/h)": "speed", "Alarm duration": "duration",
                            "Operation line": "line", "Alarm type": "alarm_type",
                            "Alarm start time": "start"})
    n_missing_driver = df["Driver"].isna().sum()
    df = df.dropna(subset=["Driver"])
    df["Driver"] = df["Driver"].astype(int)

    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()

    df["start"] = pd.to_datetime(df["start"])
    df["date"] = df["start"].dt.normalize()
    df["hour"] = df["start"].dt.hour
    df["weekend"] = df["start"].dt.dayofweek >= 5
    # Operation line: "301.0" (float artefact) -> "301"; missing -> NaN
    df["line"] = df["line"].astype(str).str.replace(r"\.0$", "", regex=True).replace({"nan": np.nan})
    df["duration"] = df["duration"].clip(upper=60)           # a few 100+ s outliers
    df["alarm_type"] = df["alarm_type"].str.strip()
    df = df[df["alarm_type"].isin(ALARM_TYPES)]

    print(f"  alarms: {n0:,} raw rows -> {len(df):,} kept "
          f"(dropped {n_missing_driver:,} with no driver ID, {n_dupes:,} exact duplicates)")
    return df.reset_index(drop=True)


def load_accidents():
    a = pd.read_excel(DATA_DIR / ACCIDENT_FILE)
    a.columns = [re.sub(r"\s+", " ", c).strip() for c in a.columns]
    flag_col = [c for c in a.columns if c.startswith("Duplicate")][0]

    # Dates are mixed: real datetimes AND text like "20-May-2024 (Mon)"
    d = a["Date of Accident"].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)
    a["acc_date"] = pd.to_datetime(d, format="mixed", errors="coerce")

    # Employee numbers: ints, plus a few like "ATO 50328" / "DA 21466"
    a["Driver"] = pd.to_numeric(a["Employee No"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    a["flag"] = a[flag_col].fillna("").astype(str).str.strip().str.upper()
    n0 = len(a)
    a = a[~a["flag"].isin(EXCLUDE_ACCIDENT_FLAGS)]
    a = a.dropna(subset=["Driver", "acc_date"])
    a["Driver"] = a["Driver"].astype(int)
    print(f"  accidents: {n0} records -> {len(a)} valid "
          f"(removed duplicates / near-misses). Date range {a.acc_date.min().date()} to {a.acc_date.max().date()}")
    return a[["BOCC Report", "acc_date", "Driver", "Service No", "flag"]]


def make_labels(acc, drivers):
    if LABEL_MODE == "march":
        m = acc.acc_date.between("2024-03-01", "2024-03-31")
    elif LABEL_MODE == "q1":
        m = acc.acc_date.between("2024-01-01", "2024-03-31")
    elif LABEL_MODE == "future":
        m = acc.acc_date > "2024-03-31"
    else:
        m = pd.Series(True, index=acc.index)
    counts = acc[m].groupby("Driver").size()
    y = pd.Series(drivers).map(counts).fillna(0).astype(int)
    print(f"  label mode '{LABEL_MODE}': {int((y > 0).sum())} of {len(y)} drivers had >=1 accident "
          f"({(y > 0).mean():.1%})")
    return y.values


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING  (one row per driver)
# ----------------------------------------------------------------------------
def build_features(df):
    g = df.groupby("Driver")
    feat = pd.DataFrame({
        "n_alarms": g.size(),
        "active_days": g["date"].nunique(),
        "n_lines": g["line"].nunique(),
        "mean_speed": g["speed"].mean(),
        "p90_speed": g["speed"].quantile(0.9),
        "mean_duration": g["duration"].mean(),
    })
    # exposure proxy: distinct (date, hour) slots in which the driver triggered an alarm
    slots = df.assign(slot=df["date"].dt.strftime("%Y%m%d") + "_" + df["hour"].astype(str))
    feat["active_hours"] = slots.groupby("Driver")["slot"].nunique()

    # alarm counts by type
    ct = df.pivot_table(index="Driver", columns="alarm_type", values="start", aggfunc="count", fill_value=0)
    for t in ALARM_TYPES:
        feat["cnt_" + slug(t)] = ct[t] if t in ct else 0

    # rates per active hour (comparable across drivers working different amounts)
    for t in ALARM_TYPES:
        feat["rate_" + slug(t)] = feat["cnt_" + slug(t)] / feat["active_hours"]
    for grp, types in GROUPS.items():
        feat[f"cnt_{grp}"] = sum(feat["cnt_" + slug(t)] for t in types)
        feat[f"rate_{grp}"] = feat[f"cnt_{grp}"] / feat["active_hours"]
    feat["alarms_per_hour"] = feat["n_alarms"] / feat["active_hours"]
    feat["alarms_per_day"] = feat["n_alarms"] / feat["active_days"]

    # context shares
    feat["night_share"] = g["hour"].apply(lambda h: ((h <= 5) | (h >= 22)).mean())     # 22:00-05:59
    feat["peak_share"] = g["hour"].apply(lambda h: h.isin([7, 8, 17, 18, 19]).mean())
    feat["weekend_share"] = g["weekend"].mean()
    feat["highspeed_share"] = g["speed"].apply(lambda s: (s >= 40).mean())
    feat["long_alarm_share"] = g["duration"].apply(lambda s: (s >= 8).mean())

    # fatigue alarms specifically at night
    fat = df[df.alarm_type.isin(GROUPS["fatigue"])]
    fat_night = fat.groupby("Driver")["hour"].apply(lambda h: ((h <= 5) | (h >= 22)).sum())
    feat["fatigue_night_cnt"] = fat_night.reindex(feat.index).fillna(0)

    # most-used line
    feat["main_line"] = g["line"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)

    feat["low_exposure"] = feat["active_days"] < MIN_ACTIVE_DAYS
    return feat.reset_index()


MODEL_FEATURES = (
    [f"rate_{slug(t)}" for t in ALARM_TYPES]
    + ["alarms_per_hour", "alarms_per_day", "active_days", "n_lines", "mean_speed", "p90_speed",
       "mean_duration", "night_share", "peak_share", "weekend_share", "highspeed_share",
       "long_alarm_share"]
)


# ----------------------------------------------------------------------------
# 3. EXPLAINABLE RISK SCORE (unsupervised)
# ----------------------------------------------------------------------------
# Weights are a judgement call - justify / tune them in your report.
SCORE_WEIGHTS = {
    "rate_harsh_braking": 1.0, "rate_harsh_acceleration": 1.0, "rate_sharp_cornering": 1.0,
    "rate_speeding": 1.5, "rate_swaying": 1.5,
    "rate_fatigue_alert": 2.0, "rate_yawn": 1.5,
    "rate_headway_monitoring_alert": 1.5, "rate_forward_collision_warning": 2.0,
    "rate_pedestrian_collision_alert": 2.5,
}


def risk_score(feat):
    """Weighted average of percentile ranks (0-100) of each alarm rate. Robust to the heavy skew."""
    pct = pd.DataFrame({c: feat[c].rank(pct=True) * 100 for c in SCORE_WEIGHTS})
    w = pd.Series(SCORE_WEIGHTS)
    score = (pct * w).sum(axis=1) / w.sum()
    return score


# ----------------------------------------------------------------------------
# 4. MACHINE LEARNING
# ----------------------------------------------------------------------------
def recall_at_top(y, p, frac=0.10):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    return y[idx].sum() / max(1, y.sum())


def train_models(feat, y):
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer

    X = feat[MODEL_FEATURES].fillna(0).values
    y = np.asarray(y)
    if y.sum() < 10:
        print("  too few positives to train a model - change LABEL_MODE")
        return None

    models = {
        "Logistic Regression": make_pipeline(
            FunctionTransformer(np.log1p), StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    # baselines to beat
    baselines = {
        "Baseline: total alarms": feat["n_alarms"].values.astype(float),
        "Baseline: alarms/hour": feat["alarms_per_hour"].values,
        "Baseline: rule-based score": feat["risk_score"].values,
        "Baseline: active days (exposure)": feat["active_days"].values.astype(float),
    }

    rows, oof = [], {}
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    for name, model in models.items():
        aucs, aps, rcs = [], [], []
        for tr, te in rskf.split(X, y):
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
            rcs.append(recall_at_top(y[te], p))
        rows.append({"model": name, "roc_auc": np.mean(aucs), "roc_auc_sd": np.std(aucs),
                     "pr_auc": np.mean(aps), "recall_top10pct": np.mean(rcs)})
        # out-of-fold probabilities -> honest per-driver predictions
        oof[name] = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                      method="predict_proba")[:, 1]
    for name, s in baselines.items():
        rows.append({"model": name, "roc_auc": roc_auc_score(y, s), "roc_auc_sd": np.nan,
                     "pr_auc": average_precision_score(y, s), "recall_top10pct": recall_at_top(y, s)})
    metrics = pd.DataFrame(rows)
    metrics["positive_rate"] = y.mean()   # PR-AUC of a random model equals this

    # feature importance (fit on everything; for interpretation only)
    from sklearn.inspection import permutation_importance
    rf = models["Random Forest"].fit(X, y)
    pi = permutation_importance(rf, X, y, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": MODEL_FEATURES, "rf_permutation_importance": pi.importances_mean})
    lr = models["Logistic Regression"].fit(X, y)
    imp["logreg_coefficient"] = lr[-1].coef_[0]
    imp = imp.sort_values("rf_permutation_importance", ascending=False)

    best = metrics[metrics.model.isin(models)].sort_values("roc_auc", ascending=False).iloc[0]["model"]
    print(f"  best model by CV ROC-AUC: {best}")
    return metrics, imp, oof, best


# ----------------------------------------------------------------------------
# 5. CHARTS FOR THE REPORT
# ----------------------------------------------------------------------------
def make_charts(df, feat, metrics, imp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4))
    df["alarm_type"].value_counts().sort_values().plot.barh(ax=ax, color="#c0392b")
    ax.set_title("Alarm volume by type (March 2024)"); ax.set_xlabel("Alarms")
    fig.tight_layout(); fig.savefig(OUT_DIR / "chart_alarm_types.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    df.groupby("hour").size().plot(ax=ax, marker="o", color="#2c3e50")
    ax.set_title("Alarms by hour of day"); ax.set_xlabel("Hour"); ax.set_ylabel("Alarms")
    fig.tight_layout(); fig.savefig(OUT_DIR / "chart_alarms_by_hour.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    feat["risk_score"].plot.hist(bins=40, ax=ax, color="#7f8c8d")
    ax.set_title("Distribution of driver risk score"); ax.set_xlabel("Risk score (0-100)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "chart_risk_score_hist.png"); plt.close(fig)

    if "had_accident" in feat:
        fig, ax = plt.subplots(figsize=(7, 4))
        feat.groupby(pd.qcut(feat["risk_score"], 10, labels=False, duplicates="drop"))["had_accident"].mean().mul(100).plot.bar(ax=ax, color="#c0392b")
        ax.set_title("Accident rate by risk-score decile (0 = lowest risk)")
        ax.set_xlabel("Risk-score decile"); ax.set_ylabel("% drivers with an accident")
        fig.tight_layout(); fig.savefig(OUT_DIR / "chart_accident_rate_by_decile.png"); plt.close(fig)
    if imp is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.head(12).iloc[::-1].plot.barh(x="feature", y="rf_permutation_importance", ax=ax, legend=False, color="#2980b9")
        ax.set_title("Top features (Random Forest permutation importance)")
        fig.tight_layout(); fig.savefig(OUT_DIR / "chart_feature_importance.png"); plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    print("[1/5] Loading data")
    raw = load_alarms()
    alarms = clean_alarms(raw)
    del raw
    acc = load_accidents()

    print("[2/5] Building driver features")
    feat = build_features(alarms)
    print(f"  {len(feat)} drivers, {len(feat.columns)} columns")

    print("[3/5] Rule-based risk score")
    feat["risk_score"] = risk_score(feat)

    print("[4/5] Labels + ML models")
    y = make_labels(acc, feat["Driver"].values)
    feat["n_accidents"] = y
    feat["had_accident"] = (y > 0).astype(int)
    res = train_models(feat, y > 0)
    imp = None
    if res:
        metrics, imp, oof, best = res
        feat["ml_risk_prob"] = oof[best]
        feat["ml_risk_pct"] = feat["ml_risk_prob"].rank(pct=True) * 100
        feat["combined_score"] = 0.5 * feat["risk_score"].rank(pct=True) * 100 + 0.5 * feat["ml_risk_pct"]
        metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)
        imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
        print("\n" + metrics.round(3).to_string(index=False) + "\n")
    else:
        feat["combined_score"] = feat["risk_score"]

    cut = feat.loc[~feat.low_exposure, "combined_score"].quantile(1 - TOP_PCT_AT_RISK)
    feat["risk_tier"] = np.where(feat["combined_score"] >= cut, "High",
                          np.where(feat["combined_score"] >= feat["combined_score"].quantile(0.7), "Medium", "Low"))
    feat.loc[feat.low_exposure, "risk_tier"] = "Insufficient data"

    print("[5/5] Saving outputs")
    feat.sort_values("combined_score", ascending=False).to_csv(OUT_DIR / "driver_risk_scores.csv", index=False)
    # compact alarm table for the dashboard (hourly x type x line), avoids shipping 4M rows
    hourly = (alarms.groupby(["date", "hour", "alarm_type", "line"], dropna=False).size().rename("n").reset_index())
    hourly.to_csv(OUT_DIR / "alarm_summary.csv", index=False)
    acc.to_csv(OUT_DIR / "accidents_clean.csv", index=False)
    make_charts(alarms, feat, res[0] if res else None, imp)

    print(f"Done in {time.time() - t0:.0f}s. Outputs in ./{OUT_DIR}/")
    return feat


if __name__ == "__main__":
    main()
