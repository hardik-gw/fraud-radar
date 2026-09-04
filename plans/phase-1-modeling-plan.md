# Phase 1 — Data & Modeling Foundation · Execution Plan

**Goal:** not "a model that works". A **documented comparison** that shows why the chosen model and
threshold were chosen. This is the phase an interviewer actually probes.

**Definition of done (from the artifact):**
> A written comparison table with real numbers for all three models, and one paragraph justifying
> the final choice and threshold.

**My execution time:** ~2–4 h. Training itself is minutes — 284,807 rows is small. The time goes
into evaluation and the write-up, which is the part with value.

---

## 1. The one idea this phase rests on

Fraud is **0.1727%** of the data. That single fact invalidates the default way of doing everything:

| Default habit | Why it fails here | What we do instead |
|---|---|---|
| Report accuracy | "Never fraud" scores 99.83% | Precision, recall, F1, **PR-AUC** |
| Random train/test split | Can leave too few frauds in test | **Stratified** split, plus a time-based hold-out |
| ROC-AUC | Looks great while drowning analysts in false positives | PR-AUC, which punishes them |
| Pick threshold 0.5 | Arbitrary, and wrong for rare events | Choose from the PR curve, **with a stated reason** |

Everything below is downstream of that table.

---

## 2. What I will build

### Step 1 — `notebooks/01-eda.ipynb`
Look before modeling. Answers, with plots:
- How is fraud distributed across the 48 hours? (Does time-of-day carry signal?)
- Which of `V1`–`V28` separate the classes most? (Ranked by class-conditional mean difference)
- What does the `Amount` distribution look like per class, on a log scale?
- Are the PCA features correlated with each other? (They shouldn't be — that's what PCA guarantees.
  Confirming it is a good check that we understand what we're holding.)

### Step 2 — `fraud_radar/` package
Small, importable Python module. **This is a deliberate addition to the artifact's plan.**

Reason: Phase 2's API must preprocess an incoming transaction *exactly* the way training did. If that
logic lives only in a notebook, Phase 2 has to copy-paste it, and the copy drifts. Training/serving
skew of this kind is one of the most common real ML bugs. So:

```
fraud_radar/
  data.py        load the CSV, stratified + time-based splits
  models.py      the three models behind one interface
  autoencoder.py PyTorch autoencoder + reconstruction-error scoring
  evaluate.py    metrics, PR curves, threshold selection
```

Preprocessing lives inside a scikit-learn **`Pipeline`**, so the scaler is saved *with* the model in
one artifact. The API can't then forget to scale.

### Step 3 — train three models
| Model | Type | Why it's here |
|---|---|---|
| Isolation Forest | Unsupervised | Learns what normal looks like; flags outliers. No labels used in training. |
| Autoencoder (PyTorch) | Unsupervised | Learns to reconstruct normal transactions; fraud reconstructs badly. Score = reconstruction error. |
| Logistic Regression + XGBoost | Supervised | The honest baseline. If they crush the unsupervised models, we say so. |

The unsupervised models train on **legitimate transactions only** — that's what "learn normal" means.

### Step 4 — evaluate all of them identically
Same test set, same metrics: precision, recall, F1, PR-AUC, plus the confusion matrix at the chosen
threshold. Results written to `models/metrics.json` so the README table is generated from real
numbers, not retyped.

### Step 5 — choose the threshold, and write down why
The genuinely interesting decision. At 1-in-578 base rate, a threshold that catches 90% of fraud may
flag thousands of legitimate customers. I'll show the tradeoff curve and pick a point framed in
business terms — *"catch X% of fraud at the cost of reviewing Y legitimate transactions per 10,000"*
— not by maximising F1 and calling it done.

### Step 6 — save the artifact
`models/fraud_model_v1.joblib` with a version tag and its metrics alongside. Gitignored; regenerated
by the training code.

### Step 7 — write it up in the README
Fill the comparison table, and write the "why not just use the labels?" paragraph with the real
numbers behind it.

---

## 3. What you have to do

**Nothing blocking.** No new accounts, no credentials. I have everything I need.

**Optional but recommended — read before the results land**, so the comparison means something:
the Fraud Detection Handbook's chapter on performance metrics (linked below). Roughly 30 minutes.
It is the single best explanation of why PR-AUC is the right call here.

**Watch for:** I'm adding PyTorch and XGBoost as dependencies. PyTorch is a large download.

---

## 4. Concepts in this phase — read & watch

### Class imbalance and why accuracy lies
- 📖 [Fraud Detection Handbook — Ch. 4: Performance metrics](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/Introduction.html) — **the priority read for this phase**
- 📖 [Google ML Crash Course — Classification: accuracy, precision, recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

### Precision / recall / PR curves
- 📺 [StatQuest — ROC and AUC, clearly explained](https://www.youtube.com/watch?v=4jRBRDbJemM) — watch this first for the intuition
- 📖 [scikit-learn — Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) — explains directly why PR beats ROC under imbalance

### Isolation Forest
- 📺 [StatQuest — Isolation Forest](https://www.youtube.com/watch?v=fLpTFsK0hqE)
- 📖 [scikit-learn — IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

### Autoencoders for anomaly detection
- 📺 [StatQuest — Autoencoders, clearly explained](https://www.youtube.com/watch?v=CS4cs9xVecg)
- 📖 [Keras — Timeseries anomaly detection with an autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection/) — Keras, but the reconstruction-error idea is identical

### Gradient boosting (the supervised baseline)
- 📺 [StatQuest — XGBoost Part 1: Regression](https://www.youtube.com/watch?v=OtD8wVaFm6E)
- 📖 [XGBoost — Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)

### Preventing training/serving skew
- 📖 [scikit-learn — Pipelines](https://scikit-learn.org/stable/modules/compose.html) — why the scaler must be saved *with* the model

---

## 5. What Phase 2 inherits

| Phase 1 output | Phase 2 uses it for |
|---|---|
| `models/fraud_model_v1.joblib` | loaded once at API startup |
| The `Pipeline` wrapping the scaler | scoring an incoming transaction without re-implementing preprocessing |
| The chosen threshold | the `flagged` boolean in the `/score` response |
| The version tag | the `model_version` field in the response |
| `fraud_radar/` package | imported directly by the API — no copy-paste |
