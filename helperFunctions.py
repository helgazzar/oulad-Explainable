# helperFunctions to support the pipeline
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pathlib import Path

if TYPE_CHECKING:
    import pipeline


def _resolve(filename: str, cfg: "pipeline.PreprocessConfig") -> Path:
    """Find a CSV across the configured search directories."""
    # test each candidate directory in priority order
    for d in cfg.data_dirs:
        # build the full candidate path
        candidate = d / filename
        # return on the first match
        if candidate.exists():
            return candidate
    # fail an empty frame
    raise FileNotFoundError(f"{filename} not found in {[str(d) for d in cfg.data_dirs]}")


def _ordinal(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Map a categorical series to ordinal codes, preserving NaN for unmapped values."""
    # .map leaves unmatched labels as NaN, which the imputer handles
    return series.map(mapping).astype("float32")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields 0.0 instead of inf/NaN on a zero denominator."""
    # np.where evaluates the whole vector at once rather than per row
    return pd.Series(
        np.where(denominator > 0, numerator / denominator.replace(0, np.nan), 0.0),
        index=numerator.index,
    ).fillna(0.0).astype("float32")

