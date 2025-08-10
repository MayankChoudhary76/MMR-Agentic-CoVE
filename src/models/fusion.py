# src/models/fusion.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def _safe_array(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None:
        return None
    if isinstance(a, list):
        a = np.array(a, dtype=np.float32)
    return a.astype(np.float32, copy=False)


def _pad_to(a: np.ndarray, dim: int) -> np.ndarray:
    """Right‑pad a to target dim with zeros if needed."""
    if a.shape[1] == dim:
        return a
    if a.shape[1] < dim:
        pad = np.zeros((a.shape[0], dim - a.shape[1]), dtype=a.dtype)
        return np.concatenate([a, pad], axis=1)
    # if larger, trim
    return a[:, :dim]


def concat_fusion(
    Vt: np.ndarray,
    Vi: Optional[np.ndarray] = None,
    Vm: Optional[np.ndarray] = None,
    weights: Tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Concatenate available modality vectors (text, image, meta).
    Optionally scale each modality by a weight before concat.
    Returns L2‑normalized fused matrix.
    """
    Vt = _safe_array(Vt)
    Vi = _safe_array(Vi)
    Vm = _safe_array(Vm)

    if weights is None:
        wt, wi, wm = 1.0, 1.0, 1.0
    else:
        wt, wi, wm = weights

    parts = []
    if Vt is not None:
        parts.append(wt * Vt)
    if Vi is not None:
        parts.append(wi * Vi)
    if Vm is not None:
        parts.append(wm * Vm)

    if not parts:
        raise ValueError("At least one modality must be provided to concat_fusion")

    fused = np.concatenate(parts, axis=1)
    return _l2norm(fused)


def weighted_sum_fusion(
    Vt: np.ndarray,
    Vi: Optional[np.ndarray] = None,
    Vm: Optional[np.ndarray] = None,
    weights: Tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Weighted sum across modalities. If dims differ, pad/trim to the
    **max** dimensionality among provided modalities before summation.
    Returns L2‑normalized fused matrix.
    """
    Vt = _safe_array(Vt)
    Vi = _safe_array(Vi)
    Vm = _safe_array(Vm)

    if weights is None:
        wt, wi, wm = 1.0, 1.0, 1.0
    else:
        wt, wi, wm = weights

    mats = []
    dims = []
    if Vt is not None:
        mats.append(("t", wt, Vt)); dims.append(Vt.shape[1])
    if Vi is not None:
        mats.append(("i", wi, Vi)); dims.append(Vi.shape[1])
    if Vm is not None:
        mats.append(("m", wm, Vm)); dims.append(Vm.shape[1])

    if not mats:
        raise ValueError("At least one modality must be provided to weighted_sum_fusion")

    target_dim = max(dims)
    acc = None
    for _, w, M in mats:
        Mpad = _pad_to(M, target_dim)
        acc = (w * Mpad) if acc is None else (acc + w * Mpad)

    return _l2norm(acc)