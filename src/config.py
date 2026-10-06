from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "logreg_reviews.joblib"
PARAMS_PATH = ROOT / "models" / "best_params.joblib"
DATA_PATH = ROOT / "data" / "Womens Clothing E-Commerce Reviews.csv"     
TARGET = "Recommended IND"
NUM_COLS = ["Age", "Positive Feedback Count"]
ONEHOT_COLS = ["Division Name", "Department Name", "Class Name"]
TEXT_COLS = ["Review Text"]