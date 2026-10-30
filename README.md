# Customer Churn Prediction & Retention Analytics

An interactive, local analytics dashboard built with **Python + Streamlit** for a
Predictive Analytics mini-project. It explores the public **IBM Telco Customer
Churn** sample dataset, demonstrates transparent data cleaning, interactive EDA,
and compares **Logistic Regression** with **Linear Discriminant Analysis (LDA)**
using leakage-safe evaluation and a real-time classification-threshold control.

> **Core question:** can customer account, service, contract, and billing
> attributes help identify customers who are more likely to churn — and how do
> Logistic Regression and LDA compare on this task?

---

## 1. Features and dashboard sections

| Section | What you can do there |
|---|---|
| **Overview** | Dataset status, headline observed-churn metrics, and the churn distribution chart. |
| **Data Explorer** | Interactive EDA: churn by any category (contract, payment method, …), tenure/charge distributions, box plots, correlation heatmap, data preview. |
| **Data Quality & Cleaning** | Raw-vs-cleaned report: missing values before/after, whitespace blanks, `TotalCharges` conversion, zero-tenure corrections, duplicates, outlier box plots, and the cleaning-actions table. |
| **Model Lab** | Train and compare Logistic Regression and LDA on the same stratified split. Move the **threshold slider (0.05–0.95)** and watch validation metrics, confusion matrix, and flagged-customer counts update instantly. Optional Logistic Regression `C` control and IQR-capping mode. |
| **Insights & Limitations** | Data-derived findings (association language only), Logistic Regression coefficient interpretation, model comparison, and responsible-use notes. |

Key evaluation behavior:

- **Validation / Interactive** results respond to the threshold slider. Probabilities
  are precomputed; moving the slider only re-binarizes predictions, so it never
  retrains the models and **ROC-AUC never moves** (it is probability-based).
- **Test / Final Evaluation** is held out, uses a fixed threshold of **0.50**, and
  is displayed separately. It is never used for threshold or model tuning.

---

## 2. Dataset

The app uses the public **IBM Telco Customer Churn sample dataset**
(7,043 rows × 21 columns, fictional telecom customers, `Churn` = target).

- Dataset page: <https://www.ibm.com/docs/en/cognos-analytics/11.2.x?topic=samples-telco-customer-churn>
- CSV URL: <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>

**Placement:** save the CSV as `data/Telco-Customer-Churn.csv`. The app loads the
local file first; if it is missing, it offers to download the documented public
source once (with clear error handling if the network is unavailable). It never
substitutes fabricated data. The raw file is never modified — cleaning produces a
separate in-memory cleaned copy.

Licensing note: the dataset is IBM's public sample data intended for demos and
education; keep it out of source control (`data/*.csv` is git-ignored).

**Get the code:**

```bash
git clone https://github.com/Mohak0204/ChurnPredictor.git
cd ChurnPredictor
```

Then follow §4 for environment setup. The dataset is downloaded automatically
on first launch (or save it manually as described above).

---

## 3. Technology stack

- **Python 3.x** — application and ML logic
- **Streamlit** — local interactive dashboard
- **Pandas / NumPy** — data handling and numeric operations
- **scikit-learn** — splitting, preprocessing, Logistic Regression, LDA, metrics
- **Plotly** — interactive charts
- **pytest** — automated tests

---

## 4. Setup (VS Code, local virtual environment)

Open the project folder in VS Code, then:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the dataset at `data/Telco-Customer-Churn.csv` (see §2), then run:

```bash
streamlit run app.py
```

The dashboard opens at <http://localhost:8501> in your browser.

---

## 5. Project structure

```text
customerchurnprediction/
├── app.py                    # Streamlit entry point (small router)
├── requirements.txt
├── .streamlit/config.toml    # Light theme tokens
├── data/                     # Telco CSV goes here (not committed)
├── src/
│   ├── config.py             # Constants: paths, schema, splits, palette
│   ├── data_loader.py        # Cached CSV loading + schema validation
│   ├── data_cleaning.py      # Raw→cleaned transformation + quality report
│   ├── eda.py                # Chart-ready summaries (within-group churn rates)
│   ├── preprocessing.py      # X/y, stratified 70/15/15 split, sklearn pipelines
│   ├── modeling.py           # Logistic Regression + LDA fitting, probabilities
│   ├── evaluation.py         # Threshold conversion + all metrics
│   ├── insights.py           # Cautious, data-derived statements
│   ├── charts.py             # Reusable Plotly chart builders
│   ├── app_state.py          # Cached split/models/evaluations for Streamlit
│   └── ui/                   # Presentation modules (one per page + layout)
└── tests/                    # pytest suite for the core logic
```

Non-UI modules are importable and testable without launching Streamlit; UI
modules only render results computed by `src/`.

---

## 6. Modeling and evaluation approach

1. **Target:** `Churn`, encoded `Yes = 1` (positive class) / `No = 0`.
   `customerID` is excluded from features; `Churn` never enters `X`.
2. **Split:** one stratified **70% train / 15% validation / 15% test** partition
   (two sequential stratified splits, fixed seed 42). Both models share exactly
   the same rows.
3. **Preprocessing:** a `ColumnTransformer` pipeline — median imputation +
   standard scaling for numeric features; most-frequent imputation + one-hot
   encoding (`handle_unknown="ignore"`) for categorical features. All
   transformations are **fitted on training data only** and applied unchanged
   to validation/test. Optional IQR capping learns its bounds from the training
   partition only.
4. **Models:** Logistic Regression (`max_iter=1000`, tunable `C`) and LDA
   (`lsqr` solver with automatic shrinkage, stable for the one-hot feature
   space). Both produce positive-class probabilities aligned to the label arrays.
5. **Metrics:** accuracy, precision, recall, F1 (`zero_division=0`), ROC-AUC
   (from probabilities), confusion matrix, predicted churn-risk count/rate.

Because the classes are imbalanced (~27% churn), accuracy alone is misleading;
F1, recall, and ROC-AUC are highlighted in the app.

### Threshold behavior

A customer is flagged as churn-risk when predicted churn probability ≥ threshold.

- **Lowering** the threshold flags more customers → recall usually rises,
  precision usually falls (more false alarms).
- **Raising** it flags fewer customers → precision usually rises, recall falls.
- ROC-AUC is computed from the probability scores and is **independent of the
  threshold** — the app demonstrates this live.

---

## 7. Testing

```bash
python -m pytest tests/ -v
```

The suite covers: schema validation, whitespace/`TotalCharges` handling, the
zero-tenure correction, duplicate accounting, raw-data immutability,
`customerID`/`Churn` exclusion, deterministic stratified splits, train-only
preprocessing fits, probability validity, coefficient extraction, confusion-
matrix orientation, known metric values, threshold invariance of ROC-AUC, and
no-positive-prediction edge cases.

---

## 8. Responsible use and limitations

- The dataset is a public educational sample for a fictional telecom company,
  not live customer data; it is a static snapshot, not a forecast of a future month.
- Predictions are **risk estimates, not certainties** about individual customers.
- Observed associations (e.g., contract type and churn) do **not** prove
  causation; the data contains no intervention outcomes, so retention savings
  cannot be claimed.
- Demographic attributes (gender, senior-citizen status) raise fairness concerns
  and must not drive unfair treatment.
- Model quality depends on the sample, features, split, and chosen threshold.

---

## 9. Known limitations / deliberate scope choices

- Single-process local app; no database, backend API, authentication, Docker,
  deep learning, or additional ML models — per the project scope.
- The seed is fixed (42) for reproducible classroom results rather than exposed
  as a control.
- Streamlit reruns re-render the page on each interaction; caching keeps data
  loading and model training from repeating, so threshold changes are instant.
