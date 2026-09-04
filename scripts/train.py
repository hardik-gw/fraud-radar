"""Train and compare every candidate model, then save the winner.

Run with:  uv run python scripts/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_radar import data, evaluate
from fraud_radar.features import ALL_FEATURES, build_features
from fraud_radar.models import build_models, make_preprocessor

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_VERSION = "v1"

# The unsupervised models learn what "normal" looks like; they do not need all
# 1.28M legitimate rows to do it, and MLPRegressor in particular gets slow well
# before it gets better. A fixed sample keeps training minutes long instead of
# hours, and the comparison stays fair because both see the same sample.
UNSUPERVISED_SAMPLE = 300_000

# Transactions a review team can look at per 10,000 processed. Every model is
# scored at this same budget so the comparison is like for like.
REVIEW_BUDGET_PER_10K = 50.0


def prepare() -> pd.DataFrame:
    """Load, engineer features, and cache the result."""
    cache = PROCESSED_DIR / "features.parquet"
    if cache.exists():
        print(f"loading cached features from {cache.relative_to(ROOT)}")
        return pd.read_parquet(cache)

    print("loading raw data ...")
    df = data.load_all()
    print(f"  {len(df):,} transactions")

    print("building features ...")
    started = time.time()
    df = build_features(df)
    print(f"  done in {time.time() - started:.0f}s")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache, index=False)
        print(f"  cached to {cache.relative_to(ROOT)}")
    except Exception as exc:  # noqa: BLE001 - the cache is an optimisation; any
        # failure to write it (missing pyarrow, read-only disk) must not stop training
        print(f"  (not cached: {exc})")
    return df


def main() -> None:
    df = prepare()

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    X_train, y_train = train[ALL_FEATURES], train["is_fraud"].to_numpy()
    X_test, y_test = test[ALL_FEATURES], test["is_fraud"].to_numpy()

    print(
        f"\ntrain {len(X_train):,} rows ({y_train.mean():.4%} fraud)"
        f"   test {len(X_test):,} rows ({y_test.mean():.4%} fraud)"
    )

    # Fit preprocessing once, on training data only. Fitting it on everything
    # would leak the test period's distribution into the scaler.
    print("\nfitting preprocessor ...")
    preprocessor = make_preprocessor().fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print(f"  {X_train_t.shape[1]} features after encoding")

    legit_idx = np.flatnonzero(y_train == 0)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        legit_idx, size=min(UNSUPERVISED_SAMPLE, len(legit_idx)), replace=False
    )
    X_normal = X_train_t[sample_idx]

    results, curves, fitted = [], {}, {}

    for name, spec in build_models().items():
        print(f"\n=== {name} ===")
        estimator = spec["estimator"]
        started = time.time()

        if spec["supervised"]:
            estimator.fit(X_train_t, y_train)
            scores = (
                estimator.predict_proba(X_test_t)[:, 1]
                if hasattr(estimator, "predict_proba")
                else estimator.decision_function(X_test_t)
            )
        else:
            # Unsupervised: fit on legitimate transactions only. This is what
            # "learn what normal looks like" means in practice.
            estimator.fit(X_normal)
            if name == "isolation_forest":
                # score_samples returns *higher = more normal*; negate so that
                # every model in this comparison uses higher = more suspicious.
                scores = -estimator.score_samples(X_test_t)
            else:
                scores = estimator.score_samples(X_test_t)

        elapsed = time.time() - started
        summary = evaluate.summarise(name, y_test, scores, REVIEW_BUDGET_PER_10K)
        summary["train_seconds"] = round(elapsed, 1)
        summary["supervised"] = spec["supervised"]
        results.append(summary)
        curves[name] = evaluate.pr_curve_points(y_test, scores)
        fitted[name] = estimator

        print(
            f"  PR-AUC {summary['pr_auc']:.4f}"
            f"   precision {summary['precision']:.3f}"
            f"   recall {summary['recall']:.3f}"
            f"   ({elapsed:.0f}s)"
        )

    results.sort(key=lambda r: r["pr_auc"], reverse=True)
    best = results[0]
    print(f"\nbest by PR-AUC: {best['model']} ({best['pr_auc']:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "version": MODEL_VERSION,
        "model_name": best["model"],
        "pipeline": Pipeline(
            [("preprocess", preprocessor), ("model", fitted[best["model"]])]
        ),
        "threshold": best["threshold"],
        "features": ALL_FEATURES,
        "metrics": best,
    }
    path = MODELS_DIR / f"fraud_model_{MODEL_VERSION}.joblib"
    joblib.dump(artifact, path)
    print(f"saved {path.relative_to(ROOT)}")

    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "review_budget_per_10k": REVIEW_BUDGET_PER_10K,
                "test_rows": len(y_test),
                # np.int64 is not JSON-serialisable, so this cast is load-bearing
                "test_fraud": int(y_test.sum()),
                "results": results,
                "pr_curves": curves,
            },
            indent=2,
        )
    )
    print("saved models/metrics.json")


if __name__ == "__main__":
    main()
