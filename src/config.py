"""Shared constants for the Customer Churn Prediction & Retention Analytics app.

Owns stable configuration values only: paths, schema, split defaults,
thresholds, and visual tokens. No data processing or model logic lives here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

# Public IBM Telco Customer Churn sample dataset.
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATASET_SOURCE_LABEL = "IBM Telco Customer Churn sample dataset"
DATASET_SOURCE_PAGE = (
    "https://www.ibm.com/docs/en/cognos-analytics/11.2.x"
    "?topic=samples-telco-customer-churn"
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

REQUIRED_COLUMNS = [
    ID_COLUMN,
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COLUMN,
]

# Columns that are candidate model predictors, by type.
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_FEATURES = NUMERIC_COLUMNS  # tenure, MonthlyCharges, TotalCharges

EXPECTED_TARGET_LABELS = ["Yes", "No"]
POSITIVE_LABEL = "Yes"  # churn is the positive class
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0

# ---------------------------------------------------------------------------
# Splits and modeling defaults
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_SEED = 42
TRAIN_FRACTION = 0.70          # of the full dataset
VALIDATION_FRACTION = 0.50     # of the remaining 30% (=> 15% overall)

DEFAULT_THRESHOLD = 0.50
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

LOGREG_MAX_ITER = 1000
LOGREG_DEFAULT_C = 1.0
LOGREG_C_OPTIONS = [0.01, 0.1, 1.0, 10.0]

# ---------------------------------------------------------------------------
# Visual tokens (restrained white / charcoal / blue / green system)
# ---------------------------------------------------------------------------
COLOR_TEXT = "#111827"        # charcoal headings/body
COLOR_TEXT_MUTED = "#64748B"  # secondary text
COLOR_BORDER = "#E2E8F0"      # light separators
COLOR_BG = "#FFFFFF"
COLOR_BG_SOFT = "#F8FAFC"
COLOR_ACCENT_BLUE = "#2563EB"
COLOR_ACCENT_GREEN = "#16A34A"
COLOR_DARK_SURFACE = "#0F172A"
COLOR_WARNING = "#D97706"

# Chart series encoding — consistent across the whole app.
COLOR_CHURN_YES = "#DC2626"     # churned customers (red = risk)
COLOR_CHURN_NO = "#2563EB"      # retained customers (blue)
CHURN_COLOR_MAP = {"Yes": COLOR_CHURN_YES, "No": COLOR_CHURN_NO}

PLOTLY_LAYOUT_TEMPLATE = {
    "template": "plotly_white",
    "font": {"family": "Inter, 'Segoe UI', sans-serif", "color": COLOR_TEXT},
    "title": {"font": {"size": 15, "color": COLOR_TEXT}},
    "margin": {"l": 50, "r": 20, "t": 55, "b": 45},
    "legend": {"orientation": "h", "y": -0.15},
    "hoverlabel": {"bgcolor": "white", "font": {"color": COLOR_TEXT}},
}
