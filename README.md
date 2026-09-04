# Fraud Radar

A real-time transaction-scoring pipeline: an anomaly-detection model trained on real credit card
fraud data, served behind an API, and fed by a live Kafka stream. The same shape as a production
fraud system like Stripe Radar, built end to end.

> **Status:** Phase 0 complete. Under active construction — see [Build phases](#build-phases).

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

[Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) —
284,807 transactions from European cardholders over two days in September 2013.

**492 of them are fraud: 0.172%.**

That number drives every modeling decision in this repo. A model that predicts "never fraud" is
99.83% accurate and catches nothing, so accuracy is not reported anywhere here. Evaluation uses
precision, recall, F1, and PR-AUC.

Features `V1`–`V28` are PCA components (the originals are confidential). `Time` and `Amount` are raw.
`Class` is the label: 1 = fraud.

The CSV is ~150 MB and is **not** in this repository — see setup below.

---

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
plans/            per-phase execution plans
```

---

## Model comparison

*Populated in Phase 1. Three models, same metrics, and a written justification for the one chosen
and the threshold it runs at.*

| Model | Type | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Isolation Forest | Unsupervised | — | — | — | — |
| Autoencoder | Unsupervised | — | — | — | — |
| Logistic Regression / XGBoost | Supervised | — | — | — | — |

### Why not just use the labels?

*Written up in Phase 1.* Short version: a supervised classifier can only recognise fraud patterns it
has already seen labelled. Real fraud systems face new patterns continuously, with labels arriving
days later via chargebacks — which is the actual argument for anomaly detection. The supervised model
here is the honest baseline the unsupervised approach has to justify itself against.

---

## Build phases

- [x] **0 · Setup & scoping** — repo, environment, dataset
- [ ] **1 · Data & modeling foundation** — EDA, three models, evaluation under class imbalance
- [ ] **2 · Serving layer** — FastAPI `/score`, `/health`, structured logging, tests
- [ ] **3 · Streaming layer** — Kafka + Postgres via Docker Compose
- [ ] **4 · Observability & explainability** — Streamlit dashboard, latency stats, SHAP
- [ ] **5 · Packaging** — one-command startup, architecture diagram, final write-up

---

## Licence / attribution

Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via
Kaggle. Methodology follows
[Reproducible Machine Learning for Credit Card Fraud Detection](https://fraud-detection-handbook.github.io/fraud-detection-handbook/).
