import logging
from pathlib import Path


def setup_logging(cfg):
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("deeponet")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(f"{cfg.log_dir}/training.log")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s"))
        logger.addHandler(fh)
    return logger
