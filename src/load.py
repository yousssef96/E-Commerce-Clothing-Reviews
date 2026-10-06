import joblib
import sklearn

from src.config import MODEL_PATH
from src.logger import logger


def load_model():
    if not MODEL_PATH.exists():
        logger.error("Model file not found: {}", MODEL_PATH)
        raise FileNotFoundError(MODEL_PATH)

    logger.info("Loading model from {} (scikit-learn {})", MODEL_PATH, sklearn.__version__)
    model = joblib.load(MODEL_PATH)
    logger.success("Model loaded")
    return model