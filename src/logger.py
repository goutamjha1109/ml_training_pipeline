import logging
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"  # always project root/logs
LOG_FILE = LOG_DIR / "training.log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# file handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)