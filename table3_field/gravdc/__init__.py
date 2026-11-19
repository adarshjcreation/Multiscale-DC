"""Inference code for 'Multiscale Learning for Downward Continuation of Gravity Data' (field data, Table 3)."""
from .data import load_test_set
from .metrics import evaluate, format_summary, save_result, summarize
from .physics import MultiscaleOperator, upward_continue
from .precision import set_precision
from .weights import load_model

set_precision("fp32")  # full FP32 + deterministic cuDNN, so results are reproducible (see precision.py)
