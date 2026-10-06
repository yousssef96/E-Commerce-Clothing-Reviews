from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from src.config import NUM_COLS, ONEHOT_COLS, TEXT_COLS
from src.preprocessing import Winsorize, clean_text_column


def build_pipeline() -> Pipeline:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", Winsorize()),
        ("scale", RobustScaler()),
    ])
    onehot = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    text = Pipeline([
        ("clean_text", FunctionTransformer(clean_text_column)),
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000,
                                  ngram_range=(1, 2), min_df=5)),
    ])
    preprocess = ColumnTransformer([
        ("num_pipeline", num, NUM_COLS),
        ("onehot_pipeline", onehot, ONEHOT_COLS),
        ("text_clean_pipeline", text, TEXT_COLS),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(solver="liblinear", max_iter=3000, random_state=42)),
    ])