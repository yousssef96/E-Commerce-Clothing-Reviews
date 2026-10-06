import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MODEL_PATH, PARAMS_PATH, TARGET
from src.logger import logger
from src.pipeline import build_pipeline


def main():
    logger.info("Loading data from {}", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded {} rows, {} columns", *df.shape)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    logger.info("Split: train={} test={} | positive rate train={:.3f}",
                len(X_train), len(X_test), y_train.mean())

    params = joblib.load(PARAMS_PATH)
    logger.info("Using best params: {}", params)

    model = build_pipeline().set_params(**params)
    logger.info("Fitting pipeline...")
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, model.predict(X_test))
    logger.success("Test AUC={:.4f} F1={:.4f}", auc, f1)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    logger.success("Model saved to {}", MODEL_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training failed")
        raise