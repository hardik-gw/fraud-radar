# Fraud Radar

A real-time transaction-scoring pipeline: an anomaly-detection model trained on real credit card
fraud data, served behind an API, and fed by a live Kafka stream. The same shape as a production
fraud system like Stripe Radar, built end to end.

> **Status:** Phases 0–1 complete. Under active construction — see [Build phases](#build-phases).

---

## Scope — what this is and isn't

**It is:** a working end-to-end system. Real dataset, trained and compared models, a live HTTP
scoring service, a Kafka stream feeding it, results persisted to Postgres, and a dashboard showing
throughput, latency, and why a given transaction was flagged.

**It isn't:** production traffic. The stream **replays a historical dataset in real time** — it does
not observe live payments. The dataset's features are anonymised PCA components (`V1`–`V28`), so the
explanations the dashboard surfaces are directionally real but not human-interpretable the way
"unusual merchant category" would be.

Stating that limit up front is deliberate. The engineering is the point; overselling it is not.

---

## Architecture

```
[ producer ] -> Kafka topic ------> [ consumer ] -> [ FastAPI /score ] -> [ Postgres ]
  replays        `transactions`       scores each      model artifact         scored
  the dataset                         message                                 results
                                                                                 |
                                                                          [ Streamlit ]
                                                                         live feed, p50/p95,
                                                                         flag rate, SHAP
```

*A proper diagram replaces this in Phase 5.*

---

## The dataset

[Sparkov simulated card transactions](https://www.kaggle.com/datasets/kartik2112/fraud-detection) —
**1,852,394 transactions**, 999 cardholders, 800 merchants, spanning **1 Jan 2019 to 31 Dec 2020**.
Fraud is **0.52%** overall.

The data ships already split in time: training runs to 21 Jun 2020, testing from there to year end.
Every number in this repo comes from that chronological hold-out, so the model is always predicting
forward — the only thing a deployed system can do. The fraud rate itself moves across the boundary,
from **0.579% to 0.386%**, so the target genuinely drifts.

### Why not the usual ULB `creditcard.csv`?

That dataset was the original choice and was swapped out at the start of Phase 1. Its features are
PCA components (`V1`–`V28`) with the real meanings deliberately removed, which makes two of this
project's goals impossible: explanations that read as anything but `V14 = -2.31`, and any feature
engineering over customer or merchant history. Since history features *are* production fraud
detection, that was disqualifying.

**The trade made in return:** Sparkov is simulated, ULB was real. Fraud here is cleaner and more
separable than reality, so **the metrics below are optimistic**. They are honest about this data;
they are not a claim about production performance.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Python is installed by uv;
you do not need a system Python.

```bash
git clone <this repo> && cd fraud-radar
uv sync                       # creates .venv from uv.lock — exact pinned versions
```

Then fetch the dataset (~144 MB uncompressed). No Kaggle account required:

```bash
curl -L -o data/raw/creditcardfraud.zip \
  "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
unzip -o data/raw/creditcardfraud.zip -d data/raw/ && rm data/raw/creditcardfraud.zip
```

Verify:

```bash
uv run python -c "import pandas as pd; d=pd.read_csv('data/raw/creditcard.csv'); print(d.shape, d.Class.mean())"
# expected: (284807, 31) 0.0017274845...
```

Or run [`notebooks/00-load-check.ipynb`](notebooks/00-load-check.ipynb), which asserts all of the
above and shows why accuracy is the wrong metric here.

### Troubleshooting: TLS-inspecting proxies

On a corporate network that intercepts TLS, Python downloads (`kagglehub`, `pip`, `requests`) fail
with `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`, while `curl` works
fine. The cause is that curl trusts the OS certificate store — which has the proxy's root CA — and
Python uses its own bundled `certifi` store, which doesn't.

Use the `curl` command above, or make Python use the OS store:

```bash
uv add --dev truststore
uv run python -c "import truststore; truststore.inject_into_ssl(); import kagglehub; print(kagglehub.dataset_download('mlg-ulb/creditcardfraud'))"
```

Never fix this by disabling certificate verification.

---

## Layout

```
data/raw/         downloaded dataset (gitignored)
data/processed/   derived data — raw is never modified in place (gitignored)
notebooks/        numbered exploration: 00-load-check, 01-eda, ...
api/              FastAPI scoring service            (Phase 2)
streaming/        Kafka producer + consumer          (Phase 3)
dashboard/        Streamlit live dashboard           (Phase 4)
models/           saved, version-tagged artifacts (gitignored)
tests/            pytest
docs/             architecture diagram, notes
fraud_radar/      shared library — features, models, evaluation
scripts/          train.py — trains all four models, writes models/metrics.json
plans/            per-phase execution plans
```

---

## Model comparison

Four models, one chronological test set (555,719 transactions, 2,145 frauds), identical metrics.
Every model is scored at the **same review budget** — 50 flagged transactions per 10,000, or 0.5% of
volume — so precision and recall are directly comparable: each is allowed the same number of alerts,
and the question is how much fraud it catches with them.

Accuracy is not reported anywhere. At a 0.39% fraud rate, flagging nothing scores 99.6%.

| Model | Type | PR-AUC | Precision | Recall | F1 | Caught | False alarms | Missed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Gradient boosting** | Supervised | **0.836** | 0.663 | 0.860 | 0.749 | 1,844 | 936 | 301 |
| Logistic regression | Supervised | 0.302 | 0.402 | 0.521 | 0.454 | 1,118 | 1,661 | 1,027 |
| Isolation forest | Unsupervised | 0.255 | 0.287 | 0.372 | 0.324 | 798 | 1,981 | 1,347 |
| Autoencoder | Unsupervised | 0.211 | 0.349 | 0.453 | 0.394 | 971 | 1,808 | 1,174 |

Reproduce with `uv run python scripts/train.py`; see
[`notebooks/02-model-comparison.ipynb`](notebooks/02-model-comparison.ipynb) for the PR curves.

### Chosen model and threshold

**Gradient boosting** (`HistGradientBoostingClassifier`) at a **threshold of 0.0113**.

That threshold was not chosen by maximising F1, which assumes a false alarm and a missed fraud cost
the same — they never do. It was chosen by fixing what a review team can absorb and reading off the
consequences. At 50 flags per 10,000 transactions, the model **catches 86.0% of fraud**, and an
analyst working that queue finds real fraud **66.3%** of the time. The cost is 301 frauds missed and
936 customers needlessly queried. Moving the threshold trades those against each other, and that
trade is a business decision, not a modelling one.

`HistGradientBoostingClassifier` rather than XGBoost because XGBoost's macOS wheels need `libomp`
from Homebrew, which isn't available on the target machine. Scikit-learn's implementation is
equivalent for this data and removes the dependency entirely.

### Why not just use the labels?

The supervised model wins decisively — PR-AUC 0.836 against 0.255 and 0.211 — and pretending
otherwise would mean torturing the setup until the answer changed. When labels exist and test-period
fraud resembles training-period fraud, supervised learning wins. That is the expected result.

The case for the unsupervised models is narrower and worth stating precisely:

- They used **no labels at all**. In production, labels arrive weeks late via chargebacks, so a
  supervised model retrained today is learning from last month's fraud.
- They flag departures from normal, so they can in principle catch a pattern nobody has labelled yet.
- They are the only option for a new segment, product or region with no fraud history.

**What this comparison does not prove:** that the unsupervised models handle novel fraud better.
Nothing in this test period is genuinely novel — the simulator generates fraud from a fixed recipe.
Demonstrating that claim needs a test period containing a pattern absent from training, which is the
drift stretch goal and is not built yet.

The deployed artifact is therefore the gradient boosting pipeline. The autoencoder stays in the
codebase as a comparison and as the basis for the drift work.

### What the feature engineering found

Two engineered history features carry real signal, and two carry none:

| Feature | Fraud (median) | Legitimate | Verdict |
|---|---:|---:|---|
| `amt_vs_card_mean` — amount in σ above this card's own average | 2.49 | −0.19 | strong |
| `implied_kmh` — travel speed implied since the last transaction | 64.0 | 21.0 | useful |
| `home_to_merchant_km` | 78.10 | 78.22 | **dead** |
| `km_from_prev_txn` | 100.26 | 100.76 | **dead** |

The geographic features are inert because Sparkov scatters merchants around each customer at random
regardless of fraud. In real card data geography is one of the strongest available signals, so this
is a limitation of the simulation rather than a finding about fraud. They are reported rather than
quietly deleted.

## Build phases

- [x] **0 · Setup & scoping** — repo, environment, dataset
- [x] **1 · Data & modeling foundation** — EDA, four models, evaluation under class imbalance
- [ ] **2 · Serving layer** — FastAPI `/score`, `/health`, structured logging, tests
- [ ] **3 · Streaming layer** — Kafka + Postgres via Docker Compose
- [ ] **4 · Observability & explainability** — Streamlit dashboard, latency stats, SHAP
- [ ] **5 · Packaging** — one-command startup, architecture diagram, final write-up

---

## Licence / attribution

Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via
Kaggle. Methodology follows
[Reproducible Machine Learning for Credit Card Fraud Detection](https://fraud-detection-handbook.github.io/fraud-detection-handbook/).
