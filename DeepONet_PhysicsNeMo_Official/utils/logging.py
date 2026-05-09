"""
Logging utility for the DeepONet training loop.

Single-handler setup — the DeepONet trainer prints its own per-epoch
table to stdout (see train.py), so this logger only writes to the
file handler. The file lands at <cfg.log_dir>/training.log and gets
appended on every run, which keeps a continuous record across
crash-and-resume cycles on the cluster.

Counterpart to utils/logging.py in the GNN and FNO pipelines, but
slimmer: those two attach a console handler and an optional W&B
hook on top, while the DeepONet runs were never wired into W&B
and don't need the duplicate console output.
"""
import logging
from pathlib import Path


def setup_logging(cfg):
    """
    Set up the 'deeponet' logger with a single file handler.

    Returns the logger directly — call sites grab it once at startup
    and pass it around. The handler-guard below makes it safe to
    call setup_logging more than once (e.g. when a notebook re-imports
    the module) without duplicating every line in the output.
    """
    # The training log lives next to the checkpoints so a single
    # output_dir holds everything from a given run.
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("deeponet")
    logger.setLevel(logging.INFO)

    # Guard against double-attach — adding the same handler twice
    # would duplicate every log line, and the issue isn't always
    # obvious until the file is twice as big as expected.
    if not logger.handlers:
        fh = logging.FileHandler(f"{cfg.log_dir}/training.log")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s"))
        logger.addHandler(fh)
    return logger