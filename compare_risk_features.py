"""CEG2003 experiment for predicting which bus captains have later reports.

Run from the project folder with Python 3.10+ and pandas 2+, NumPy,
scikit-learn 1.2+, matplotlib and openpyxl. This script reads the data directly.

    python compare_risk_features.py
    python compare_risk_features.py --alarms alarms_prepared.csv --reports reports_prepared.csv
    python compare_risk_features.py --final-test

The default run compares methods on validation folds. Use --final-test when
the choice of features, methods and evaluation metric is final.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from math import ceil
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits


SEED = 42
BATCH_SIZE = 150_000
TOP_FRACTION = 0.10
ALARM_TYPES = (
    "Harsh Acceleration", "Harsh Braking", "Speeding", "Swaying", "Sharp Cornering",
    "Fatigue Alert", "Yawn", "Headway Monitoring Alert", "Forward Collision Warning",
    "Pedestrian Collision Alert",
)
FATIGUE_TYPES = ("Fatigue Alert", "Yawn")
COLLISION_TYPES = ("Headway Monitoring Alert", "Forward Collision Warning", "Pedestrian Collision Alert")
EXCLUDED_REPORT_FLAGS = {"DUPLICATE", "DD", "NEAR MISS", "NM", "TRUE", "YES", "1", "1.0"}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


TYPE_RATES = tuple(f"rate_{slug(name)}" for name in ALARM_TYPES)
RULE_WEIGHTS = (1, 1, 1.5, 1.5, 1, 2, 1.5, 1.5, 2, 2.5)
EXISTING_FEATURES = TYPE_RATES + (
    "alarms_per_hour", "alarms_per_day", "active_days", "n_lines", "mean_speed", "p90_speed",
    "mean_duration", "night_share", "peak_share", "weekend_share", "highspeed_share", "long_alarm_share",
)
RISK_MARKERS = {
    "high_speed_braking_rate": "Harsh braking at 40 km/h or more per 100 hours with alarms",
    "high_speed_collision_warning_rate": "Forward or pedestrian warnings at 40 km/h or more per 100 hours with alarms",
    "night_fatigue_rate": "Fatigue alerts or yawns per 100 nighttime hours with alarms",
    "night_alarm_hour_share": "Share of observed alarm hours between 22:00 and 05:59",
    "fatigue_day_share": "Share of alarm days with a fatigue alert or yawn",
    "collision_warning_day_share": "Share of alarm days with a headway, forward or pedestrian warning",
    "repeated_collision_hour_share": "Share of alarm hours with at least two headway, forward or pedestrian warnings",
    "busiest_day_alarm_share": "Share of a captain's alarms recorded on their busiest alarm day",
}

REQUIRED_ALARM_COLUMNS = {
    "Driver": ("Driver ID cleaned", "Driver cleaned", "Driver", "Driver ID"),
    "line": ("Operation line cleaned", "Operation line", "line", "route"),
    "alarm_type": ("Alarm type",),
    "start": ("Alarm start time cleaned", "Alarm start time", "start", "start time"),
    "speed": ("speed(km/h)", "speed kmh", "speed"),
    "duration": ("Alarm duration", "duration"),
}
OPTIONAL_ALARM_COLUMNS = {
    "end": ("Alarm end time cleaned", "Alarm end time", "end", "end time"),
    "conflict": ("Conflicting event details", "conflict flag", "is conflict"),
}


def column_mapping(headers, required, optional):
    lookup = {slug(name): name for name in headers}
    mapping = {}
    for target, aliases in (required | optional).items():
        source = next((lookup[slug(name)] for name in aliases if slug(name) in lookup), None)
        if source is not None:
            mapping[source] = target
    missing = [aliases[0] for name, aliases in required.items() if name not in mapping.values()]
    if missing:
        raise ValueError(f"Missing columns {', '.join(missing)}. Available columns are {list(headers)}.")
    return mapping


def alarm_batches(path):
    """Read only the fields the experiment needs. XLSX sheets are streamed."""
    if path.suffix.lower() == ".csv":
        headers = pd.read_csv(path, nrows=0).columns
        mapping = column_mapping(headers, REQUIRED_ALARM_COLUMNS, OPTIONAL_ALARM_COLUMNS)
        with pd.read_csv(path, usecols=list(mapping), dtype=str, chunksize=BATCH_SIZE) as reader:
            for chunk in reader:
                yield chunk.rename(columns=mapping)
        return
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected a CSV or XLSX alarm file, got {path}.")

    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            rows = sheet.values
            headers = next(rows, None)
            if not headers or not any(headers):
                continue
            print(f"  Reading sheet {sheet.title}", flush=True)
            mapping = column_mapping(headers, REQUIRED_ALARM_COLUMNS, OPTIONAL_ALARM_COLUMNS)
            positions = [headers.index(name) for name in mapping]
            while block := list(islice(rows, BATCH_SIZE)):
                yield pd.DataFrame.from_records(
                    (tuple(row[pos] for pos in positions) for row in block), columns=list(mapping.values())
                )
    finally:
        workbook.close()


def parse_dates(values):
    return pd.to_datetime(values, format="mixed", errors="coerce", dayfirst=True)


def driver_ids(values):
    # Examples from the file formats include 12345.0, ATO 12345 and DA 12345.
    codes = values.astype("string").str.strip().str.extract(r"(?i)^(?:[a-z]+\s*)?(\d+)(?:\.0+)?$")[0]
    return pd.to_numeric(codes, errors="coerce").astype("Int64")


def truthy(values):
    return values.astype("string").str.strip().str.lower().isin({"true", "yes", "1", "1.0"})


def clean_alarm_batch(frame, start_date, end_date, audit):
    driver = driver_ids(frame["Driver"])
    line = frame["line"].astype("string").str.strip().str.upper().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    line = line.mask(line.isin({"", "N/A", "NAN", "NONE", "NULL", "<NA>"}))
    type_lookup = {name.lower(): name for name in ALARM_TYPES}
    alarm_type = frame["alarm_type"].astype("string").str.strip().str.lower().map(type_lookup)
    start = parse_dates(frame["start"])
    speed = pd.to_numeric(frame["speed"], errors="coerce")
    duration = pd.to_numeric(frame["duration"], errors="coerce")

    valid = driver.notna() & line.notna() & alarm_type.notna() & start.notna()
    valid &= np.isfinite(speed) & speed.ge(0) & np.isfinite(duration) & duration.gt(0)
    end = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    in_window = start.ge(start_date) & start.lt(end_date)
    if "end" in frame:
        end = parse_dates(frame["end"])
        valid &= end.ge(start)
        in_window &= end.lt(end_date)
    flagged = truthy(frame["conflict"]) if "conflict" in frame else pd.Series(False, index=frame.index)
    keep = valid & ~flagged & in_window

    audit["input_rows"] += len(frame)
    audit["flagged_conflicts"] += int(flagged.sum())
    audit["invalid_rows"] += int((~flagged & ~valid).sum())
    audit["outside_window"] += int((~flagged & valid & ~in_window).sum())
    return pd.DataFrame({
        "Driver": driver[keep].astype("int64"), "line": line[keep],
        "alarm_type": pd.Categorical(alarm_type[keep], categories=ALARM_TYPES),
        "start": start[keep], "end": end[keep], "speed": speed[keep], "duration": duration[keep],
    })


def load_alarms(paths, start_date, end_date):
    audit = Counter()
    batches = []
    for path in paths:
        print(f"Reading alarms from {path}", flush=True)
        for frame in alarm_batches(path):
            alarms = clean_alarm_batch(frame, start_date, end_date, audit)
            if not alarms.empty:
                batches.append(alarms)
    if not batches:
        raise ValueError("No complete alarm rows in the feature window. Check the input columns and dates.")

    alarms = pd.concat(batches, ignore_index=True)
    del batches
    alarms["line"] = alarms["line"].astype("category")
    rows_before = len(alarms)
    alarms = alarms.drop_duplicates(ignore_index=True)
    audit["duplicate_rows"] = rows_before - len(alarms)
    audit["used_rows"] = len(alarms)
    audit["dates_with_alarms"] = alarms["start"].dt.normalize().nunique()
    return alarms, audit


def load_reports(path):
    if path.suffix.lower() == ".xlsx":
        with pd.ExcelFile(path) as workbook:
            sheet = next((name for name in workbook.sheet_names if slug(name) == "database"), 0)
            frame = workbook.parse(sheet)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Expected a CSV or XLSX report file, got {path}.")

    required = {
        "Driver": ("Employee No cleaned", "Employee No", "Employee Number", "Employee ID", "Driver"),
        "date": ("Date of Accident cleaned", "Accident date cleaned", "acc_date", "Date of Accident", "Accident Date"),
    }
    optional = {"flag": ("flag",), "duplicate": ("Marked DUPLICATE",)}
    mapping = column_mapping(frame.columns, required, optional)
    if "flag" not in mapping.values():
        flag_column = next((name for name in frame if slug(name).startswith("duplicate")), None)
        if flag_column is None:
            flag_column = next((name for name in frame if slug(name) in {"classification", "accident_classification"}), None)
        if flag_column is not None:
            mapping[flag_column] = "flag"
    frame = frame[list(mapping)].rename(columns=mapping)
    date = frame["date"].astype("string").str.strip().str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
    output = pd.DataFrame({"Driver": driver_ids(frame["Driver"]), "date": parse_dates(date)})
    keep = output["Driver"].notna() & output["date"].notna()
    if "flag" in frame:
        flags = frame["flag"].astype("string").str.strip().str.upper().str.replace(r"[\s_-]+", " ", regex=True)
        keep &= ~flags.isin(EXCLUDED_REPORT_FLAGS)
    if "duplicate" in frame:
        keep &= ~truthy(frame["duplicate"])
    return output.loc[keep], len(frame)


def is_night(hours):
    return (hours < 6) | (hours >= 22)


def build_features(alarms):
    """Summarize each captain's alarm history without referring to outcomes."""
    driver = alarms["Driver"]
    hours = alarms["start"].dt.hour
    night = is_night(hours)
    fatigue = alarms["alarm_type"].isin(FATIGUE_TYPES)
    collision = alarms["alarm_type"].isin(COLLISION_TYPES)
    high_speed = alarms["speed"].ge(40)
    high_speed_collision = high_speed & alarms["alarm_type"].isin(COLLISION_TYPES[1:])

    by_driver = alarms.groupby("Driver", sort=True, observed=True)
    counts = alarms.groupby(["Driver", "alarm_type"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=ALARM_TYPES, fill_value=0)
    signals = pd.DataFrame({
        "night_share": night, "peak_share": hours.isin([7, 8, 17, 18, 19]),
        "weekend_share": alarms["start"].dt.dayofweek.ge(5), "highspeed_share": high_speed,
        "long_alarm_share": alarms["duration"].ge(8),
    })
    feature = signals.groupby(driver).mean()
    feature["n_alarms"] = by_driver.size()
    feature["n_lines"] = by_driver["line"].nunique()
    feature["mean_speed"] = by_driver["speed"].mean()
    feature["p90_speed"] = by_driver["speed"].quantile(0.9)
    feature["mean_duration"] = alarms["duration"].clip(upper=60).groupby(driver).mean()

    # Use distinct calendar days/hours that contain alarms as exposure proxies.
    daily = pd.DataFrame({"fatigue": fatigue, "collision": collision})
    daily = daily.groupby([driver, alarms["start"].dt.normalize()], sort=False).agg(
        alarm_count=("collision", "size"), had_fatigue=("fatigue", "any"), had_collision=("collision", "any")
    ).groupby(level="Driver")
    feature["active_days"] = daily.size()
    feature["busiest_day_alarm_share"] = daily["alarm_count"].max() / feature["n_alarms"]
    feature["fatigue_day_share"] = daily["had_fatigue"].mean()
    feature["collision_warning_day_share"] = daily["had_collision"].mean()

    hourly = collision.groupby([driver, alarms["start"].dt.floor("h")], sort=False).sum()
    active_hours = hourly.groupby(level="Driver").size()
    night_slots = is_night(hourly.index.get_level_values(1).hour)
    night_hours = hourly[night_slots].groupby(level="Driver").size().reindex(feature.index, fill_value=0)
    feature["night_alarm_hour_share"] = night_hours / active_hours
    feature["repeated_collision_hour_share"] = hourly.ge(2).groupby(level="Driver").mean()
    feature["night_fatigue_rate"] = (100 * (night & fatigue).groupby(driver).sum() / night_hours.where(night_hours > 0)).fillna(0)
    high_speed_braking = high_speed & alarms["alarm_type"].eq("Harsh Braking")
    feature["high_speed_braking_rate"] = 100 * high_speed_braking.groupby(driver).sum() / active_hours
    feature["high_speed_collision_warning_rate"] = 100 * high_speed_collision.groupby(driver).sum() / active_hours
    feature["alarms_per_hour"] = feature["n_alarms"] / active_hours
    feature["alarms_per_day"] = feature["n_alarms"] / feature["active_days"]

    rates = counts.div(active_hours, axis=0)
    rates.columns = list(TYPE_RATES)
    return feature.join(rates).sort_index()


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    columns: tuple[str, ...]
    estimator: BaseEstimator | None = None


def comparison_methods():
    supervised = {
        "Logistic regression": make_pipeline(
            FunctionTransformer(np.log1p), StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=12, max_depth=6, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=2, random_state=SEED,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=20, max_iter=100, learning_rate=0.05,
            class_weight="balanced", random_state=SEED,
        ),
    }
    expanded = EXISTING_FEATURES + tuple(RISK_MARKERS)
    methods = [
        Method("Total alarms", "baseline", ("n_alarms",)),
        Method("Alarms per alarm hour", "baseline", ("alarms_per_hour",)),
        Method("Weighted alarm score", "baseline", TYPE_RATES),
        Method("Isolation forest", "unsupervised", expanded, IsolationForest(n_estimators=160, n_jobs=2, random_state=SEED)),
    ]
    for name, estimator in supervised.items():
        methods.append(Method(f"{name} / existing", "supervised", EXISTING_FEATURES, estimator))
        methods.append(Method(f"{name} / + markers", "supervised", expanded, estimator))
    return methods


def score_method(method, train, train_labels, unseen):
    train_values = train.loc[:, list(method.columns)].to_numpy()
    unseen_values = unseen.loc[:, list(method.columns)].to_numpy()
    if method.kind == "baseline":
        if unseen_values.shape[1] == 1:
            return unseen_values[:, 0]
        percentiles = []
        for reference, values in zip(train_values.T, unseen_values.T):
            reference = np.sort(reference)
            left = np.searchsorted(reference, values, side="left")
            right = np.searchsorted(reference, values, side="right")
            percentiles.append((left + right) / (2 * len(reference)))
        return np.average(percentiles, axis=0, weights=RULE_WEIGHTS)

    estimator = clone(method.estimator)
    if method.kind == "unsupervised":
        estimator.fit(train_values)
        return -estimator.score_samples(unseen_values)
    estimator.fit(train_values, train_labels)
    return estimator.predict_proba(unseen_values)[:, 1]


def ranking_metrics(labels, scores, tie_order):
    flagged = max(1, ceil(TOP_FRACTION * len(labels)))
    selected = np.lexsort((tie_order, -scores))[:flagged]
    caught = int(labels[selected].sum())
    positives = int(labels.sum())
    precision = caught / flagged
    positive_rate = positives / len(labels)
    return {
        "captains": len(labels), "positive_captains": positives, "positive_rate": positive_rate,
        "average_precision": average_precision_score(labels, scores), "roc_auc": roc_auc_score(labels, scores),
        "top10_flagged": flagged, "top10_found": caught, "top10_precision": precision,
        "top10_recall": caught / positives, "top10_lift": precision / positive_rate,
    }


def compare_on_validation(methods, features, labels):
    folds = min(5, int(np.bincount(labels, minlength=2).min()))
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=2, random_state=SEED)
    results = []
    for fold, (train, validation) in enumerate(cv.split(features, labels), 1):
        print(f"  Validation fold {fold}/{2 * folds}", flush=True)
        tie_order = np.random.default_rng(SEED + fold).permutation(len(validation))
        for method in methods:
            scores = score_method(method, features.iloc[train], labels[train], features.iloc[validation])
            metrics = ranking_metrics(labels[validation], scores, tie_order)
            results.append({"fold": fold, "method": method.name, "kind": method.kind, **metrics})
    per_fold = pd.DataFrame(results)
    comparison = per_fold.groupby(["method", "kind"], as_index=False, sort=False).agg(
        ap_mean=("average_precision", "mean"), ap_sd=("average_precision", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        top10_precision_mean=("top10_precision", "mean"), top10_recall_mean=("top10_recall", "mean"),
        top10_lift_mean=("top10_lift", "mean"), positive_rate_mean=("positive_rate", "mean"),
    ).sort_values(["ap_mean", "top10_precision_mean", "method"], ascending=[False, False, True])
    return comparison.reset_index(drop=True), per_fold


def plot_overview(plt, counts, features, labels, path):
    from matplotlib.ticker import EngFormatter

    fig, (types, volume, outcome) = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
    counts.sort_values().plot.barh(ax=types, color="#278271")
    types.set(title="Alarms by type", xlabel="Alarm events", ylabel="")
    types.xaxis.set_major_formatter(EngFormatter())
    totals = features["n_alarms"].to_numpy()
    volume.hist(totals, bins=np.geomspace(totals.min(), totals.max() + 1, 21), color="#3574ad")
    volume.set(title="Alarms per captain", xlabel="Alarm events (log scale)", ylabel="Captains", xscale="log")
    volume.xaxis.set_major_formatter(EngFormatter())
    bars = outcome.bar(["No linked report", "Linked report"], np.bincount(labels, minlength=2), color=["#3574ad", "#c05e35"])
    outcome.bar_label(bars, padding=3)
    outcome.margins(y=0.16)
    outcome.set(title="Later report labels", ylabel="Captains")
    fig.suptitle("Training pool before model fitting")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(plt, comparison, path):
    sorted_rows = comparison.iloc[::-1]
    kinds = {"baseline": "#6f7780", "supervised": "#3574ad", "unsupervised": "#278271"}
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    ax.barh(sorted_rows["method"], sorted_rows["ap_mean"], xerr=sorted_rows["ap_sd"],
            capsize=3, color=[kinds[kind] for kind in sorted_rows["kind"]])
    ax.axvline(sorted_rows["positive_rate_mean"].iloc[0], color="black", linestyle="--", label="Constant-score baseline")
    ax.set(xlabel="Average precision (mean and fold standard deviation)", title="Model comparison on identical validation folds")
    max_bar = (sorted_rows["ap_mean"] + sorted_rows["ap_sd"]).max()
    ax.set_xlim(0, max(max_bar, sorted_rows["positive_rate_mean"].iloc[0]) * 1.08)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def input_paths(args):
    data = args.data_dir
    prepared_alarms = data / "alarms_prepared.csv"
    alarms = args.alarms or ([prepared_alarms] if prepared_alarms.is_file() else [data / f"Mar24_part{i}.xlsx" for i in (1, 2)])
    report_names = ("reports_prepared.csv", "Traffic_Accident_Database_2024.xlsx", "Traffic Accident Database 2024.xlsx")
    reports = args.reports or next((data / name for name in report_names if (data / name).is_file()), data / report_names[1])
    for path in [*alarms, reports]:
        if not path.is_file():
            raise ValueError(f"File missing: {path}. Use --alarms and --reports to point to your CSV or XLSX files.")
    return alarms, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder to search for the default input files")
    parser.add_argument("--alarms", type=Path, nargs="+", help="One or more alarm CSV or XLSX files")
    parser.add_argument("--reports", type=Path, help="Accident report CSV or XLSX file")
    parser.add_argument("--out-dir", type=Path, help="Folder for the experiment results")
    parser.add_argument("--alarm-start", default="2024-03-01", help="First alarm date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--alarm-end", default="2024-03-31", help="Last alarm date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--outcome-end", default="2024-05-31", help="Last outcome date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--final-test", action="store_true",
                        help="Evaluate the strongest learned method and baseline on the reserved 20%%")
    args = parser.parse_args()

    try:
        alarm_paths, report_path = input_paths(args)
        start, last_alarm, last_report = (pd.Timestamp(s).normalize() for s in (args.alarm_start, args.alarm_end, args.outcome_end))
        if not start <= last_alarm < last_report:
            raise ValueError("Choose an alarm start and end followed by a later outcome end date.")
        outcome_start = last_alarm + pd.Timedelta(days=1)
        outcome_stop = last_report + pd.Timedelta(days=1)
        alarms, audit = load_alarms(alarm_paths, start, outcome_start)
        features = build_features(alarms)
        reports, report_rows = load_reports(report_path)
        later = reports.loc[reports["date"].ge(outcome_start) & reports["date"].lt(outcome_stop)]
        labels = features.index.isin(later["Driver"]).astype(int)
        if np.bincount(labels, minlength=2).min() < 3:
            report_range = f"{reports['date'].min():%Y-%m-%d} to {reports['date'].max():%Y-%m-%d}" if len(reports) else "empty"
            raise ValueError(
                f"Found {int(labels.sum())} captains with linked later reports out of {len(labels)}. "
                f"Usable report dates are {report_range}. Check the input reports and date window. "
                "For a March-only file, try --alarm-end 2024-03-15 --outcome-end 2024-03-31."
            )
        train, test = train_test_split(np.arange(len(labels)), test_size=0.20, stratify=labels, random_state=SEED)
        if np.bincount(labels[train], minlength=2).min() < 2 or np.bincount(labels[test], minlength=2).min() == 0:
            raise ValueError("Both outcomes need representation in the training folds and reserved test. Check the input date window.")
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError("A driver feature is missing or non-finite. Check the input data.")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Input error: {error}\n")

    training = features.iloc[train]
    train_labels = labels[train]
    output = args.out_dir or Path("risk_experiment_final" if args.final_test else "risk_experiment_outputs")
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alarm_counts = alarms.loc[alarms["Driver"].isin(training.index), "alarm_type"].value_counts()
    del alarms
    plot_overview(plt, alarm_counts, training, train_labels, output / "data_overview.png")

    methods = comparison_methods()
    print(f"\nComparing methods using {len(training):,} training captains and {int(train_labels.sum()):,} positive labels.")
    print(f"The other {len(test):,} captains are reserved for the final test.", flush=True)
    with threadpool_limits(limits=2):
        comparison, per_fold = compare_on_validation(methods, training, train_labels)

    is_baseline = comparison["kind"].eq("baseline")
    learned = comparison.loc[~is_baseline].iloc[0]
    baseline = comparison.loc[is_baseline].iloc[0]
    comparison.to_csv(output / "validation_comparison.csv", index=False)
    per_fold.to_csv(output / "validation_folds.csv", index=False)
    plot_comparison(plt, comparison, output / "validation_comparison.png")

    paired = per_fold.pivot(index="fold", columns="method", values="average_precision")
    delta_lines = []
    for name in ("Logistic regression", "Random forest", "Gradient boosting"):
        delta = paired[f"{name} / + markers"] - paired[f"{name} / existing"]
        delta_lines.append(f"  {name}  {delta.mean():+.4f} AP, higher on {int(delta.gt(0).sum())} of {len(delta)} folds")

    comparison_text = comparison[["method", "kind", "ap_mean", "ap_sd", "roc_auc_mean", "top10_precision_mean", "top10_recall_mean"]]
    report_lines = [
        "CEG2003 Bus Captain Risk Experiment", "",
        f"Alarm feature dates            {start:%Y-%m-%d} to {last_alarm:%Y-%m-%d}",
        f"Later report dates             {outcome_start:%Y-%m-%d} to {last_report:%Y-%m-%d}",
        f"Alarm input rows               {audit['input_rows']:,}",
        f"Flagged conflicting rows       {audit['flagged_conflicts']:,}",
        f"Incomplete or invalid rows     {audit['invalid_rows']:,}",
        f"Rows outside alarm window      {audit['outside_window']:,}",
        f"Duplicate events removed       {audit['duplicate_rows']:,}",
        f"Alarm rows used                {audit['used_rows']:,} across {audit['dates_with_alarms']} dates",
        f"Report rows read and usable    {report_rows:,} read, {len(reports):,} usable",
        f"Usable rows in outcome window  {len(later):,} across {later['Driver'].nunique():,} employee IDs",
        f"Captains                       {len(labels):,} with one row each",
        f"Training pool                  {len(train):,} captains, {int(train_labels.sum()):,} linked to later reports ({train_labels.mean():.2%})",
        f"Reserved final test            {len(test):,} different captains", "",
        "Positive means a captain linked to an included later report. Negative means no such report was linked.",
        "The report cleanup excludes duplicates, DD, near misses and rows with missing or invalid dates or IDs.",
        "Hours and days are distinct time slots with recorded alarms. They are proxies for observation and do not measure driving time.",
        "Alarm types become numeric counts and rates. Employee and route IDs are excluded from the model inputs.",
        f"Existing predictors per captain  {len(EXISTING_FEATURES)}. With engineered markers  {len(EXISTING_FEATURES) + len(RISK_MARKERS)}.",
        "Isolation forest fits alarm patterns without seeing report labels. The supervised models learn from those labels.", "",
        "Risk markers added", *(f"  {description}" for description in RISK_MARKERS.values()), "",
        "Validation model comparison", comparison_text.round(4).to_string(index=False), "",
        f"All methods used the same {per_fold['fold'].nunique()} validation folds. Each validation captain was outside that fold's training set.",
        "Average precision (AP) summarizes precision and recall as more captains are flagged.",
        f"Constant ranking AP  {comparison['positive_rate_mean'].iloc[0]:.4f} (average positive label rate across folds).",
        "The top 10% is rounded up to a whole captain. Precision is the share flagged with a linked report. Recall is the share of linked captains found.",
        f"Strongest learned method by validation AP  {learned['method']} ({learned['ap_mean']:.4f}).",
        f"Strongest simple baseline by validation AP  {baseline['method']} ({baseline['ap_mean']:.4f}).", "",
        "Effect of adding the engineered markers on identical folds", *delta_lines,
    ]
    if args.final_test:
        test_labels = labels[test]
        tie_order = np.random.default_rng(SEED).permutation(len(test))
        final_rows = []
        methods_by_name = {method.name: method for method in methods}
        with threadpool_limits(limits=2):
            for name in (learned["method"], baseline["method"]):
                scores = score_method(methods_by_name[name], training, train_labels, features.iloc[test])
                final_rows.append({"method": name, **ranking_metrics(test_labels, scores, tie_order)})
        final = pd.DataFrame(final_rows)
        final.to_csv(output / "final_test_results.csv", index=False)
        report_lines += ["", "Final test metrics", final.round(4).to_string(index=False)]
    else:
        report_lines += ["", "Run with --final-test after choosing the features and methods to evaluate on the reserved captains."]

    report = "\n".join(report_lines) + "\n"
    (output / "experiment_summary.txt").write_text(report, encoding="utf-8")
    print(f"\n{report}\nResults directory  {output.resolve()}")


if __name__ == "__main__":
    main()
