"""
Compare every model, show a detailed report card for EACH model, then name the best one.

Run AFTER atrisk_pipeline.py:     python compare_models.py
Reads outputs/driver_risk_scores.csv + outputs/accidents_clean.csv (takes a few minutes).

Outputs (in ./outputs):
  model_evaluation_report.txt   full report: one card per model + leaderboard + verdict
  model_comparison.csv          every number in one table
  compare_*.png                 charts (AUC bars, ROC / precision-recall, lift, importance)

How the BEST model is chosen
  Each trained model is ranked on four metrics that matter for this problem
  (ROC-AUC, PR-AUC, balanced accuracy and precision in the top 10%). The model with the
  best AVERAGE rank wins (ties broken by ROC-AUC). The report then checks whether the win
  is real: margin of error, shuffle test (with a correction for trying several models)
  and a comparison against simple untrained scores.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, permutation_test_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from atrisk_pipeline import MODEL_FEATURES

warnings.filterwarnings("ignore")
OUT = Path("outputs")
SEED = 42
LABEL_MODE = "all"        # "all", "q1", "march" or "future" (see atrisk_pipeline.py)
N_FOLDS, N_REPEATS = 5, 3
N_PERMUTATIONS = 50       # shuffle-test repeats per model (more = finer p-values, slower)
TOP_FRACS = [0.05, 0.10, 0.20, 0.30]
FLAG_FRAC = 0.20          # "flag the riskiest 20%" is used for accuracy / precision / recall / F1
W = 92

log1p = FunctionTransformer(np.log1p)


def get_models():
    return {
        "Logistic Regression": make_pipeline(log1p, StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30,
            class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
            max_iter=150, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
        "SVM (RBF)": make_pipeline(log1p, StandardScaler(),
            SVC(C=0.5, kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)),
        "K-Nearest Neighbours": make_pipeline(log1p, StandardScaler(),
            KNeighborsClassifier(n_neighbors=25, weights="distance")),
        "Naive Bayes": make_pipeline(log1p, StandardScaler(), GaussianNB()),
    }


def make_label(acc, drivers, mode):
    d = acc["acc_date"]
    mask = {"all": d.notna(), "q1": d <= "2024-03-31",
            "march": d.between("2024-03-01", "2024-03-31"), "future": d > "2024-03-31"}[mode]
    return drivers.isin(acc.loc[mask, "Driver"]).astype(int).values


def flag_top(s, frac):
    """0/1 vector flagging the riskiest `frac` of captains (rank based, so ties cannot flag everyone)."""
    k = max(1, int(round(len(s) * frac)))
    flags = np.zeros(len(s), dtype=int)
    flags[np.argsort(-s, kind="stable")[:k]] = 1
    return flags


def top_frac_stats(y, s, frac):
    flags = flag_top(s, frac)
    caught = int(y[flags == 1].sum())
    k = int(flags.sum())
    prec = caught / k
    return dict(flagged=k, caught=caught, precision=prec, recall=caught / y.sum(), lift=prec / y.mean())


def metrics_for(y, s, probability=True):
    """All decision-relevant numbers for one set of scores."""
    f20, f10 = flag_top(s, FLAG_FRAC), flag_top(s, 0.10)
    m = dict(
        auc=roc_auc_score(y, s), ap=average_precision_score(y, s),
        acc20=accuracy_score(y, f20), prec20=precision_score(y, f20, zero_division=0),
        rec20=recall_score(y, f20), f1_20=f1_score(y, f20, zero_division=0),
        bal20=balanced_accuracy_score(y, f20),
        top10_prec=y[f10 == 1].mean(), top10_lift=y[f10 == 1].mean() / y.mean())
    if probability:                       # default 50% cut-off, only meaningful for real probabilities
        p05 = (s >= 0.5).astype(int)
        m.update(acc05=accuracy_score(y, p05), prec05=precision_score(y, p05, zero_division=0),
                 rec05=recall_score(y, p05), share05=p05.mean())
    else:
        m.update(acc05=np.nan, prec05=np.nan, rec05=np.nan, share05=np.nan)
    return m


def boot_ci(y, s, n=400):
    rng = np.random.default_rng(SEED)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def line(ch="-"):
    return ch * W


def main():
    feat = pd.read_csv(OUT / "driver_risk_scores.csv")
    acc = pd.read_csv(OUT / "accidents_clean.csv", parse_dates=["acc_date"])
    X = feat[MODEL_FEATURES].fillna(0).values
    y = make_label(acc, feat["Driver"], LABEL_MODE)
    n, pos = len(y), int(y.sum())
    print(f"Label mode '{LABEL_MODE}': {pos} accident drivers out of {n} ({y.mean():.1%})")
    print(f"Testing every model with {N_REPEATS}x repeated {N_FOLDS}-fold cross-validation "
          f"(each score comes from captains the model had NOT seen).\n")

    splits = list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(X, y))
    models = get_models()

    rec, oof = {}, {}
    for name, model in models.items():
        folds, pooled = [], np.zeros(n)
        for r in range(N_REPEATS):
            fs = np.zeros(n)
            for tr, te in splits[r * N_FOLDS:(r + 1) * N_FOLDS]:
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[te])[:, 1]
                fs[te] = p
                folds.append(metrics_for(y[te], p))
            pooled += pd.Series(fs).rank(pct=True).values / N_REPEATS
        fd = pd.DataFrame(folds)
        d = fd.mean().to_dict()
        d["auc_sd"] = fd["auc"].std()
        d["kind"] = "ML model"
        d["auc_lo"], d["auc_hi"] = boot_ci(y, pooled)
        rec[name], oof[name] = d, pooled
        print(f"  tested {name:<22} AUC {d['auc']:.3f} +/- {d['auc_sd']:.3f}   "
              f"accuracy {d['acc05']:.3f}  precision {d['prec05']:.3f}", flush=True)

    # shuffle test: is each model better than luck?
    print("\nRunning the shuffle test (is each model better than luck?) ...", flush=True)
    for name, model in get_models().items():
        _, perm, pval = permutation_test_score(model, X, y, scoring="roc_auc",
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            n_permutations=N_PERMUTATIONS, random_state=SEED, n_jobs=-1)
        rec[name]["p_raw"] = pval
        rec[name]["p_adj"] = min(1.0, pval * len(models))      # Bonferroni: we tried several models
        rec[name]["perm_auc"] = perm.mean()
        print(f"  {name:<22} p = {pval:.3f}", flush=True)

    # feature importance per model (fit on all data - for interpretation only)
    importance = {}
    for name, model in get_models().items():
        model.fit(X, y)
        pi = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
        importance[name] = pd.Series(pi.importances_mean, index=MODEL_FEATURES).sort_values(ascending=False)

    # simple untrained baselines
    base = {"Rule-based alarm score": feat["risk_score"].values,
            "Alarms per hour (one number)": feat["alarms_per_hour"].values,
            "Forward-collision rate only": feat["rate_forward_collision_warning"].values,
            "Random guessing": np.random.default_rng(SEED).random(n)}
    for name, s in base.items():
        d = metrics_for(y, s, probability=False)
        d["auc_sd"] = np.nan
        d["kind"] = "Simple baseline"
        d["auc_lo"], d["auc_hi"] = boot_ci(y, s)
        rec[name] = d
        oof[name] = pd.Series(s).rank(pct=True).values

    comp = pd.DataFrame(rec).T
    num_cols = [c for c in comp.columns if c != "kind"]
    comp[num_cols] = comp[num_cols].astype(float)

    # ---------------------------------------------------------------- pick the best ML model
    ml = comp[comp.kind == "ML model"].copy()
    rank_metrics = ["auc", "ap", "bal20", "top10_prec"]
    for m in rank_metrics:
        ml["rk_" + m] = ml[m].rank(ascending=False, method="min")
    ml["avg_rank"] = ml[["rk_" + m for m in rank_metrics]].mean(axis=1)
    ml = ml.sort_values(["avg_rank", "auc"], ascending=[True, False])
    ml["rank"] = range(1, len(ml) + 1)
    best = ml.index[0]
    comp = comp.join(ml[["rank", "avg_rank"] + ["rk_" + m for m in rank_metrics]])

    # ---------------------------------------------------------------- build the report
    L = []
    P = L.append
    P(line("="))
    P("MODEL EVALUATION REPORT".center(W))
    P(line("="))
    P(f"Task      : predict which of {n:,} bus captains had an accident ({pos} did = {y.mean():.1%}).")
    P(f"Test      : {N_REPEATS}x repeated {N_FOLDS}-fold cross-validation; every score is from captains "
      f"the model had not seen.")
    P(f"Models    : {len(ml)} trained models + {len(base)} simple baselines (random guessing = the floor).")
    P(f"Label set : '{LABEL_MODE}' accidents.")
    P("")
    P("HOW TO READ THE NUMBERS")
    P("  ROC-AUC      : chance the model ranks an accident captain above a non-accident one. 0.50 = guessing.")
    P(f"  PR-AUC       : precision-recall quality. Guessing scores {y.mean():.3f} (the accident rate).")
    P(f"  Accuracy     : share of captains classified correctly. 'Predict nobody' already gets {1 - y.mean():.1%}!")
    P("  Precision    : of the captains FLAGGED as at-risk, the share who really had an accident.")
    P("  Recall       : of all accident captains, the share that were flagged.")
    P("  Lift         : precision divided by the overall accident rate (1.0x = no better than random).")

    # ---- one card per model, best first
    for name in ml.index:
        r = comp.loc[name]
        s = oof[name]
        P("")
        P(line("="))
        tag = "   <<< BEST MODEL" if name == best else ""
        P(f"MODEL {int(r['rank'])} OF {len(ml)}: {name.upper()}{tag}")
        P(line("="))

        P("HEADLINE SCORES (average over all test folds)")
        P(f"  ROC-AUC            {r.auc:.3f}   (+/- {r.auc_sd:.3f} between folds; 95% range {r.auc_lo:.3f}-{r.auc_hi:.3f})   guessing = 0.500")
        P(f"  PR-AUC             {r.ap:.3f}   guessing = {y.mean():.3f}")
        P(f"  Balanced accuracy  {r.bal20:.3f}   (flagging riskiest {FLAG_FRAC:.0%})   guessing = 0.500")

        P("")
        P("ACCURACY AND PRECISION")
        P(f"  {'':<34}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>7}{'Flagged':>9}")
        f1_05 = 2 * r.prec05 * r.rec05 / (r.prec05 + r.rec05) if (r.prec05 + r.rec05) > 0 else 0
        P(f"  {'Default 50% cut-off':<34}{r.acc05:>10.1%}{r.prec05:>11.1%}{r.rec05:>9.1%}{f1_05:>7.3f}{r.share05:>9.1%}")
        P(f"  {'Flag riskiest ' + format(FLAG_FRAC, '.0%'):<34}{r.acc20:>10.1%}{r.prec20:>11.1%}{r.rec20:>9.1%}{r.f1_20:>7.3f}{FLAG_FRAC:>9.1%}")
        P(f"  (random picking would give precision {y.mean():.1%}; predicting 'nobody' gives accuracy {1 - y.mean():.1%})")

        P("")
        P("IF WE FLAG THE RISKIEST X% OF CAPTAINS")
        P(f"  {'Flag top':<9}{'Captains':>9}{'Accident captains caught':>28}{'Precision':>11}{'Recall':>9}{'Lift':>7}")
        for f in TOP_FRACS:
            t = top_frac_stats(y, s, f)
            P(f"  {f:>7.0%}  {t['flagged']:>9}{t['caught']:>15} of {pos:<8}{t['precision']:>10.1%}{t['recall']:>9.1%}{t['lift']:>6.2f}x")

        flags = flag_top(s, FLAG_FRAC)
        tn, fp, fn, tp = confusion_matrix(y, flags).ravel()
        P("")
        P(f"CONFUSION MATRIX (flag riskiest {FLAG_FRAC:.0%}; all captains, each predicted while unseen)")
        P(f"  {'':<16}{'Flagged':>10}{'Not flagged':>14}")
        P(f"  {'Had accident':<16}{tp:>10}{fn:>14}")
        P(f"  {'No accident':<16}{fp:>10}{tn:>14}")
        P(f"  -> {tp} accident captains caught, {fn} missed, {fp} flagged without an accident.")

        dec = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=False) + 1
        rate = pd.Series(y).groupby(dec.values).mean() * 100
        P("")
        P("ACCIDENT RATE BY PREDICTED-RISK DECILE (10 = riskiest; a good model rises left to right)")
        P("  " + "  ".join(f"{d:>4}" for d in rate.index))
        P("  " + "  ".join(f"{v:>4.0f}" for v in rate.values) + f"   (% with accident; overall {y.mean() * 100:.0f}%)")

        P("")
        P("TOP MEASURES THE MODEL RELIES ON")
        for feat_name, v in importance[name].head(4).items():
            P(f"  {feat_name:<34}{v:>8.4f}")

        P("")
        P("IS IT REAL? (shuffle test: re-run with accident labels scrambled)")
        P(f"  Real AUC {r.auc:.3f} vs {r.perm_auc:.3f} with scrambled labels | p = {r.p_raw:.3f}"
          f" | p after correcting for {len(ml)} models = {r.p_adj:.3f}")
        P(f"  -> {'Better than luck' if r.p_adj < 0.05 else ('Possibly better than luck, but not proven after correction' if r.p_raw < 0.05 else 'NOT shown to be better than luck')}")

    # ---- leaderboard
    P("")
    P(line("="))
    P("LEADERBOARD: ALL MODELS SIDE BY SIDE".center(W))
    P(line("="))
    P("")
    P("TABLE 1 - HOW WELL DOES EACH MODEL RANK CAPTAINS? (used to choose the best model)")
    P(f"  {'Rank':<5}{'Model':<30}{'ROC-AUC':>8}{'95% range':>14}{'PR-AUC':>8}{'BalAcc':>8}{'Top10 prec':>11}{'Top10 lift':>11}{'Avg rank':>9}")
    for name, r in ml.iterrows():
        mark = " *" if name == best else "  "
        P(f"  {int(r['rank']):<5}{(name + mark):<30}{r.auc:>8.3f}{f'{r.auc_lo:.3f}-{r.auc_hi:.3f}':>14}{r.ap:>8.3f}"
          f"{r.bal20:>8.3f}{r.top10_prec:>10.1%} {r.top10_lift:>9.2f}x{r.avg_rank:>9.2f}")
    P(f"  {'-':<5}--- simple baselines (no training) ---")
    for name in base:
        r = comp.loc[name]
        P(f"  {'-':<5}{name:<30}{r.auc:>8.3f}{f'{r.auc_lo:.3f}-{r.auc_hi:.3f}':>14}{r.ap:>8.3f}"
          f"{r.bal20:>8.3f}{r.top10_prec:>10.1%} {r.top10_lift:>9.2f}x{'':>9}")
    P("  * = best model. Avg rank = average of the model's rank on ROC-AUC, PR-AUC, BalAcc and Top10 precision.")

    P("")
    P(f"TABLE 2 - ACCURACY AND PRECISION (default 50% cut-off, and when flagging the riskiest {FLAG_FRAC:.0%})")
    P(f"  {'Model':<30}{'Acc@50%':>9}{'Prec@50%':>10}{'Flagged@50%':>13}{'Acc@20':>9}{'Prec@20':>9}{'Recall@20':>11}{'F1@20':>7}")
    for name, r in ml.iterrows():
        mark = " *" if name == best else "  "
        P(f"  {(name + mark):<30}{r.acc05:>9.1%}{r.prec05:>10.1%}{r.share05:>13.1%}{r.acc20:>9.1%}{r.prec20:>9.1%}{r.rec20:>11.1%}{r.f1_20:>7.3f}")
    for name in base:
        r = comp.loc[name]
        P(f"  {name:<30}{'n/a':>9}{'n/a':>10}{'n/a':>13}{r.acc20:>9.1%}{r.prec20:>9.1%}{r.rec20:>11.1%}{r.f1_20:>7.3f}")
    P(f"  'Predict nobody has an accident' scores {1 - y.mean():.1%} accuracy: accuracy alone cannot pick the best model.")

    P("")
    P("TABLE 3 - IS EACH RESULT REAL? (shuffle test; p below 0.05 = unlikely to be luck)")
    P(f"  {'Model':<30}{'p-value':>9}{'Adjusted p':>12}{'AUC vs rule-based score':>26}   Verdict")
    rb = comp.loc["Rule-based alarm score", "auc"]
    for name, r in ml.iterrows():
        mark = " *" if name == best else "  "
        verdict = ("Real" if r.p_adj < 0.05 else ("Maybe" if r.p_raw < 0.05 else "Not shown"))
        P(f"  {(name + mark):<30}{r.p_raw:>9.3f}{r.p_adj:>12.3f}{r.auc - rb:>+26.3f}   {verdict}")
    P(f"  Adjusted p multiplies each p-value by {len(ml)} because {len(ml)} models were tried "
      f"(trying many models makes a lucky win likely).")

    # ---- the verdict
    b = ml.loc[best]
    wins = [lab for lab, m in [("ROC-AUC", "auc"), ("PR-AUC", "ap"), ("balanced accuracy", "bal20"),
                               ("top-10% precision", "top10_prec")] if ml[m].idxmax() == best]
    tied = [nm for nm, r in ml.iterrows() if nm != best and b.auc_lo <= r.auc <= b.auc_hi]
    best_base = comp.loc[list(base)[:3]].sort_values("auc", ascending=False).iloc[0]
    P("")
    P(line("="))
    P(f"VERDICT: THE BEST MODEL IS {best.upper()}".center(W))
    P(line("="))
    P(f"  Why it wins   : best average rank ({b.avg_rank:.2f}) across ROC-AUC, PR-AUC, balanced accuracy and "
      f"top-10% precision.")
    P(f"                  It is #1 on: {', '.join(wins) if wins else 'none of the four individually (it is the most consistent)'}.")
    P(f"  Its scores    : ROC-AUC {b.auc:.3f} | PR-AUC {b.ap:.3f} (guessing {y.mean():.3f}) | "
      f"accuracy {b.acc05:.1%} | precision {b.prec05:.1%}")
    P(f"                  Flag the riskiest 10%: {b.top10_prec:.1%} had an accident = {b.top10_lift:.2f}x the random rate.")
    P("")
    P("  How sure are we?")
    P(f"   - Shuffle test      : p = {b.p_raw:.3f}; after correcting for {len(ml)} models p = {b.p_adj:.3f} "
      f"-> {'real' if b.p_adj < 0.05 else ('suggestive but not proven' if b.p_raw < 0.05 else 'not shown to be real')}.")
    P("   - Margin of error   : " + (f"{', '.join(tied)} score within its 95% range, so the ranking among them "
      f"is NOT reliable (treat as roughly tied)." if tied else "no other model falls inside its 95% range."))
    P(f"   - Versus simple     : best untrained score ({best_base.name}) gets AUC {best_base.auc:.3f} vs "
      f"{b.auc:.3f} here -> " + ("the model adds little over a simple alarm measure."
      if b.auc - best_base.auc < 0.03 else "the model clearly beats simple scores."))
    P(f"   - Accuracy caveat   : accuracy ({b.acc05:.1%}) is below the 'predict nobody' baseline ({1 - y.mean():.1%}) "
      f"because the model flags captains on purpose; judge it by precision, recall and AUC instead.")
    report = "\n".join(L) + "\n"
    (OUT / "model_evaluation_report.txt").write_text(report, encoding="utf-8")
    out = comp.reset_index().rename(columns={"index": "model"})
    out["positive_rate"] = y.mean()
    out.to_csv(OUT / "model_comparison.csv", index=False)
    print("\n" + report)

    # ---------------------------------------------------------------- charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    c = comp.sort_values("auc")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if m == best else ("#95a5a6" if k == "Simple baseline" else "#2980b9")
              for m, k in zip(c.index, c.kind)]
    ax.barh(c.index, c.auc, color=colors, xerr=[c.auc - c.auc_lo, c.auc_hi - c.auc],
            error_kw=dict(ecolor="black", lw=1, capsize=3))
    ax.axvline(0.5, color="k", ls="--"); ax.set_xlim(0.4, 0.7)
    ax.set_xlabel("ROC-AUC (0.5 = guessing)"); ax.set_title("Model comparison (red = best, bars = 95% range)")
    fig.tight_layout(); fig.savefig(OUT / "compare_auc_bars.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name in list(ml.index[:4]) + ["Rule-based alarm score"]:
        fpr, tpr, _ = roc_curve(y, oof[name]); ax[0].plot(fpr, tpr, label=name)
        pr, rc, _ = precision_recall_curve(y, oof[name]); ax[1].plot(rc, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].set_title("ROC curves"); ax[0].legend(fontsize=7)
    ax[1].axhline(y.mean(), color="k", ls="--"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-recall curves")
    fig.tight_layout(); fig.savefig(OUT / "compare_roc_pr.png"); plt.close(fig)

    dec = pd.qcut(pd.Series(oof[best]).rank(method="first"), 10, labels=False) + 1
    rate = pd.Series(y).groupby(dec.values).mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(rate.index, rate.values, color="#c0392b"); ax.axhline(y.mean() * 100, color="k", ls="--", label="Overall")
    ax.set_xlabel("Predicted-risk decile (10 = riskiest)"); ax.set_ylabel("% with accident"); ax.set_title(f"Lift by decile - {best}"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "compare_best_lift.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance[best].head(10).iloc[::-1].plot.barh(ax=ax, color="#2980b9")
    ax.set_title(f"Top measures used by {best}")
    fig.tight_layout(); fig.savefig(OUT / "compare_best_importance.png"); plt.close(fig)
    print("Saved model_evaluation_report.txt, model_comparison.csv and compare_*.png in ./outputs/")


if __name__ == "__main__":
    main()
