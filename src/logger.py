import sys
from loguru import logger

from src.config import ROOT

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()   # drop the default handler so messages aren't duplicated

logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}",
)
logger.add(
    LOG_DIR / "app.log",
    level="DEBUG",
    rotation="5 MB",       # start a new file at 5 MB
    retention="14 days",   # delete old files
    compression="zip",
    enqueue=True,          # safe with uvicorn workers
    backtrace=True,
    diagnose=False,        # don't dump variable values (could include review text) into logs
)

__all__ = ["logger"]