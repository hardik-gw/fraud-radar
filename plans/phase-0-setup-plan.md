# Phase 0 — Setup & Scoping · Execution Plan

> **Naming note:** you said "phase one", and in the Fraud Radar artifact the *first* phase is
> labelled **Phase 0 · Setup & scoping**. That's what this document covers. The artifact's
> **Phase 1 · Data & modeling foundation** is the next one — it can't start until this is done
> (no repo, no environment, no dataset = nothing to model). Say the word and I'll write that plan too.

**Goal of this phase:** get the ground under us before a single line of modeling code.
Repo, reproducible environment, dataset in hand, README skeleton.

**Realistic time:** ~45–90 min of my execution + ~15 min of your clicking. Not a full day.

**Definition of done (from the artifact):**
> Repo scaffolded, dataset loads locally, environment installs cleanly on a second machine.

---

## 0. Starting state of your machine

I checked before planning. Facts, not assumptions:

| Thing | Status | Consequence |
|---|---|---|
| `/Users/hardik.k/Desktop/fraud radar` | empty, **not a git repo** | I run `git init` |
| `python3` | **3.9.6** (Apple's system Python) | Too old & shouldn't be touched. We install our own. |
| `pip3` | 21.2.4 (ancient, system) | Won't be used directly |
| `git` | 2.50.1 ✅ | Fine |
| `docker` | **not installed** | Needed in Phase 3, **not now**. Flagged, not blocking. |
| `brew`, `uv`, `poetry`, `kaggle` | not installed | We install `uv` only |

**Why we don't just use the system Python 3.9:** macOS ships it for its *own* tooling. Installing
project packages into it breaks OS scripts, and 3.9 is old enough that some current ML wheels won't
resolve. Standard practice is: never touch the system interpreter, install your own per project.

---

## 1. Decisions I'm making (and why)

You should be able to defend each of these in an interview, so here's the reasoning, not just the choice.

### Decision 1 — `uv` instead of `venv` + `pip` or Poetry
**What it is:** `uv` is a single fast tool that installs Python versions, creates virtual
environments, and resolves/locks dependencies.

**Why:** it solves our exact problem in one step — it can *install Python 3.11 itself*, so you don't
need Homebrew or a python.org installer. It also writes a `uv.lock` file, which is what actually
makes "installs cleanly on a second machine" true rather than aspirational. `requirements.txt` alone
records what you asked for, not the exact resolved versions of every transitive dependency.

**Fallback:** if you'd rather stick to the built-in tooling, plain `python3 -m venv` +
`requirements.txt` works and I'll do that instead — just tell me. The rest of the plan is unchanged.

### Decision 2 — Python 3.11
Broadest wheel compatibility across scikit-learn / XGBoost / PyTorch / Kafka clients. 3.12 would
almost certainly work too; 3.11 removes a class of "no matching distribution" annoyance for zero cost.

### Decision 3 — dataset stays out of git
`creditcard.csv` is ~150 MB. GitHub's hard limit is 100 MB per file. It goes in `.gitignore`, and the
README documents how to fetch it. This is the normal convention — **data is not source code.**

### Decision 4 — folder layout
The artifact specifies `/data`, `/notebooks`, `/api`, `/streaming`, `/dashboard`. I'm adding
`/models`, `/tests`, `/docs` because later phases need them and retrofitting directories mid-project
produces messy commit history. Layout is loosely
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) — a widely recognised
convention, so a reviewer knows where to look without asking.

### Decision 5 — `data/raw` vs `data/processed`
Raw downloaded data is **never** modified in place. Anything we derive gets written to
`data/processed/`. This means every transformation is reproducible from the original file, which is
the whole point of the word "reproducible" in the handbook we're following.

---

## 2. What I will do — step by step

Each step lists the actual command or file so you can follow along rather than watch a black box.

### Step 1 — Initialise the repository
```bash
git init
git branch -M main
```
Creates `.git/` and sets the default branch to `main`.

### Step 2 — Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Installs a single binary to `~/.local/bin/uv`. No system Python is touched.
*(This downloads and runs a script from the internet — it's the vendor's official installer, but I'll
show it to you before running it. If you'd prefer, I'll use the `venv` fallback instead.)*

### Step 3 — Pin the Python version and create the project
```bash
uv python install 3.11
uv init --python 3.11
uv venv
```
Produces `pyproject.toml` (declares the project + its dependencies) and `.venv/` (the isolated
environment). `.python-version` records the interpreter so a second machine gets the same one.

### Step 4 — Add Phase 0 + Phase 1 dependencies
```bash
uv add pandas numpy scikit-learn matplotlib seaborn jupyter joblib
uv add --dev ruff pytest
```
Deliberately **not** installing FastAPI / Kafka / Torch yet — those belong to the phases that use
them. Adding everything up front makes it impossible to tell which phase needs what, and slows every
install. Writes `uv.lock` with exact pinned versions.

### Step 5 — Create the directory tree
```
data/raw/          # creditcard.csv lands here — gitignored
data/processed/    # anything we derive — gitignored
notebooks/         # exploration, numbered: 01-eda.ipynb, 02-...
api/               # Phase 2 (FastAPI service)
streaming/         # Phase 3 (Kafka producer/consumer)
dashboard/         # Phase 4 (Streamlit)
models/            # saved model artifacts — gitignored
tests/             # pytest
docs/              # architecture diagram, notes
plans/             # these plan documents
```
Each empty directory gets a `.gitkeep` — git tracks files, not folders, so empty dirs vanish otherwise.

### Step 6 — Write `.gitignore`
Covers `.venv/`, `__pycache__/`, `data/`, `models/*.pkl`, `.ipynb_checkpoints/`, `.env`,
`.DS_Store`, `kaggle.json`.

**`kaggle.json` and `.env` are the important two** — those hold credentials. Committing an API key to
a public GitHub repo is the single most common way people leak secrets, and scrapers find them within
minutes. We block it structurally rather than by remembering.

### Step 7 — Fetch the dataset
Preferred (after your Step B below):
```bash
uv run kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```
Fallback if the Kaggle CLI gives you trouble: you download the ZIP in the browser and I'll unpack it
into `data/raw/`. Either is fine — this is a one-time fetch, not part of the pipeline.

### Step 8 — Verification notebook `notebooks/00-load-check.ipynb`
Small on purpose. It proves the environment works and gives you your first real look at the data:
- `df = pd.read_csv("data/raw/creditcard.csv")` → assert shape is **(284807, 31)**
- `df["Class"].value_counts(normalize=True)` → confirm fraud rate ≈ **0.172%** (492 of 284,807)
- `df.isnull().sum().sum()` → expect 0
- `df[["Time", "Amount"]].describe()` → note the raw scales, which is *why* Phase 1 scales them
- print column list → V1–V28 are anonymised PCA components; `Time`, `Amount`, `Class` are not

That 0.172% number is the entire reason this project is interesting. Accuracy is a useless metric
here: a model that predicts "never fraud" scores **99.83% accurate** and catches zero fraud.

### Step 9 — README skeleton
Sections: problem statement · architecture (placeholder for the Phase 5 diagram) · dataset + how to
get it · setup instructions · **scope note** ("replays historical data in real time; this is not live
production traffic") · phase-by-phase status checklist.

The scope note is not modesty — overstating what a portfolio project does is a fast way to lose
credibility in an interview. Stating the limit yourself reads as judgment.

### Step 10 — First commit
```bash
git add -A
git commit -m "Phase 0: scaffold repo, environment, dataset load check"
```
One clean commit. Phase 5's definition of done includes "readable commit history", which starts here.

---

## 3. What **you** have to do

Only three things need a human. Everything else is mine.

### A. Decide on `uv` — 1 min
Confirm Decision 1 above, or say "use plain venv" and I'll switch. **Blocking** — everything follows
from it.

### B. Get a Kaggle API token — ~10 min
1. Create/log into an account at [kaggle.com](https://www.kaggle.com)
2. Open the dataset page: **[Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** and click to accept its terms — the API won't download it until you have
3. Go to [kaggle.com/settings/api](https://www.kaggle.com/settings/api) → **Create New Token** → downloads `kaggle.json`
4. Move it into place and lock it down:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   `chmod 600` = only you can read it. The Kaggle CLI actually warns you if it's more permissive.

**Do not paste the contents of `kaggle.json` into this chat.** It's a live credential. Put it in
`~/.kaggle/` yourself; I never need to see it.

*Blocking for Step 7 only — I can do Steps 1–6 and 9 while you handle this.*

### C. Read/watch the concept list in §5 — ~40 min
The point of this project is that you understand it. This phase's concepts are the cheapest ones to
learn, so it's worth doing now rather than later.

### Not needed yet
- **Docker Desktop** — Phase 3. Install it before then; it's a big download, so starting it early
  isn't wasted, but nothing here needs it.
- **A GitHub repo** — we commit locally now, push in Phase 5 when there's something worth showing.
  Say so if you'd rather push from day one.

---

## 4. How you verify it worked

Run these yourself when I say I'm done. If all four pass, Phase 0 is genuinely complete:

```bash
uv run python -c "import pandas, sklearn; print(pandas.__version__, sklearn.__version__)"
uv run python -c "import pandas as pd; d=pd.read_csv('data/raw/creditcard.csv'); print(d.shape, d.Class.mean())"
git log --oneline
git status --short          # data/ and .venv/ must NOT appear
```

Expected: versions print · `(284807, 31) 0.001727...` · one commit · clean status.

The last one is the real test — if `creditcard.csv` shows up in `git status`, `.gitignore` is wrong.

---

## 5. Concepts in this phase — read & watch

Short list on purpose. These are the ideas Phase 0 actually uses.

### Virtual environments — *why isolate at all?*
The single concept that makes "runs on my machine" reproducible.
- 📺 [Python Virtual Environments - Full Tutorial for Beginners](https://www.youtube.com/watch?v=Y21OR1OPC9A)
- 📺 [Python Virtual Environments Explained (venv)](https://www.youtube.com/watch?v=G9_FNnApn_E) — shorter
- 📖 [Real Python — Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/) — the best written explanation of *why*, not just how

### `uv` — the tool we're using
- 📺 [UV - A Faster, All-in-One Package Manager to Replace Pip and Venv](https://www.youtube.com/watch?v=AMdG7IjgSPM)
- 📖 [uv — Getting started](https://docs.astral.sh/uv/getting-started/) · [Working on projects](https://docs.astral.sh/uv/guides/projects/)

### Git — enough to keep clean history
You don't need branching yet; you need init/add/commit/log and to understand what `.gitignore` does.
- 📺 [Corey Schafer — Git Tutorial for Beginners: Command-Line Fundamentals](https://www.youtube.com/watch?v=HVsySz-h9r4) — the canonical one
- 📖 [gitignore.io](https://www.toptal.com/developers/gitignore) — generates ignore files per stack

### Project structure for ML work
- 📖 [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) — read the
  "Directory structure" and the opinions page. This is where the raw/processed split comes from.

### The dataset itself
- 📖 [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) —
  read the description tab. Note the line recommending **AUPRC** over accuracy; that's Phase 1's
  whole evaluation strategy in one sentence.
- 📖 [Fraud Detection Handbook — Ch. 2: ML for credit card fraud detection](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_2_Background/MachineLearningForFraudDetection.html) —
  the artifact's recommended reading. **Chapter 2 only for now**; Chapter 4 (performance metrics)
  is Phase 1 homework.

### Optional, if you want the "why this project" framing
- 📖 [How Stripe Radar works](https://stripe.com/radar/guide) — the real system whose shape we're
  borrowing. Useful vocabulary for talking about the project.

---

## 6. What Phase 1 inherits from this

So you can see why each piece exists rather than treating setup as ritual:

| Phase 0 output | Phase 1 uses it for |
|---|---|
| `data/raw/creditcard.csv` | EDA, stratified train/test split |
| `.venv` with pandas/sklearn | every model in the comparison table |
| `models/` + `.gitignore` rules | saving the chosen model artifact with a version tag |
| `notebooks/` numbering convention | `01-eda.ipynb`, `02-baselines.ipynb`, … stay ordered |
| The 0.172% fraud rate you saw in Step 8 | the reason we report PR-AUC instead of accuracy |
| README scope note | the "why not just use the labels?" paragraph gets appended to it |

---

## 7. Open questions for you

1. **`uv` or plain `venv`+`requirements.txt`?** (Recommend `uv`.)
2. **Push to GitHub now, or at Phase 5?** (Recommend Phase 5 — but if you want a public commit
   history from day one, now is the time to create the repo.)
3. **Do you want me to execute, or do you want to type the commands yourself while I explain?**
   You said you don't need to write the code — but Phase 0 is short and hands-on-keyboard here
   builds real muscle memory for the rest. Your call.
