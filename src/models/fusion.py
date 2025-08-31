# src/models/fusion.py
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Tuple

__all__ = ["l2norm", "concat_fusion", "weighted_sum_fusion"]


def _as_float32(x: np.ndarray | Sequence) -> np.ndarray:
    """Cast to contiguous float32 ndarray (2D if possible)."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, arr.shape[-1])
    return np.ascontiguousarray(arr, dtype=np.float32)


def l2norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    L2-normalize a 2D array [N, D] along the last dimension.

    Parameters
    ----------
    x : np.ndarray
        Array of shape [N, D].
    eps : float
        Small number to avoid division by zero.

    Returns
    -------
    np.ndarray
        L2-normalized array (float32) of shape [N, D].
    """
    x = _as_float32(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms


def _gather_modalities(
    Vt: Optional[np.ndarray] = None,
    Vi: Optional[np.ndarray] = None,
    Vm: Optional[np.ndarray] = None,
) -> Tuple[list[np.ndarray], list[str]]:
    """Collect non-None modality matrices in consistent order."""
    blocks, names = [], []
    if Vt is not None:
        blocks.append(_as_float32(Vt)); names.append("text")
    if Vi is not None:
        blocks.append(_as_float32(Vi)); names.append("image")
    if Vm is not None:
        blocks.append(_as_float32(Vm)); names.append("meta")
    if not blocks:
        raise ValueError("At least one modality (Vt/Vi/Vm) must be provided.")
    # Sanity: same item count across blocks
    n = {b.shape[0] for b in blocks}
    if len(n) != 1:
        raise ValueError(f"All modalities must have same N items, got Ns={n}.")
    return blocks, names


def concat_fusion(
    Vt: Optional[np.ndarray] = None,
    Vi: Optional[np.ndarray] = None,
    Vm: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Concatenate modality vectors (optionally weighted per block) and L2-normalize.

    Parameters
    ----------
    Vt, Vi, Vm : np.ndarray or None
        Matrices of shape [I, Dt], [I, Di], [I, Dm].
    weights : sequence of floats or None
        Optional per-modality weights in the order (text, image, meta).
        Missing modalities are skipped; when None, all used modalities get weight=1.

    Returns
    -------
    np.ndarray
        Fused item matrix of shape [I, Dt+Di+Dm_used], L2-normalized.
    """
    blocks, names = _gather_modalities(Vt, Vi, Vm)

    # Build weights aligned to available modalities
    if weights is None:
        w = [1.0] * len(blocks)
    else:
        # Map full tuple -> subset by names present
        name_order = ["text", "image", "meta"]
        full = dict(zip(name_order, list(weights) + [1.0] * (3 - len(weights))))
        w = [float(full[n]) for n in names]

    # Apply weights per block
    weighted = [b * w_i for b, w_i in zip(blocks, w)]

    fused = np.concatenate(weighted, axis=1)
    return l2norm(fused)


def _pad_to_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Zero-pad vectors to target_dim along the last axis.

    Parameters
    ----------
    x : [N, D]
    target_dim : int

    Returns
    -------
    [N, target_dim]
    """
    x = _as_float32(x)
    N, D = x.shape
    if D == target_dim:
        return x
    if D > target_dim:
        # If any block is larger than target, bump target up
        raise ValueError(f"Block has dim {D} > target_dim {target_dim}.")
    out = np.zeros((N, target_dim), dtype=np.float32)
    out[:, :D] = x
    return out


def weighted_sum_fusion(
    Vt: Optional[np.ndarray] = None,
    Vi: Optional[np.ndarray] = None,
    Vm: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Weighted sum fusion across modalities with automatic zero-padding to a common dim.

    Notes
    -----
    - This keeps a simple, dependency-free baseline. We align dimensions by zero-padding
      smaller blocks to the maximum dimensionality across provided modalities, then sum.
    - Returned vectors are L2-normalized.

    Parameters
    ----------
    Vt, Vi, Vm : np.ndarray or None
        Matrices of shape [I, Dt], [I, Di], [I, Dm].
    weights : sequence of floats or None
        Per-modality weights in order (text, image, meta). If None → weights=1.

    Returns
    -------
    np.ndarray
        Fused item matrix of shape [I, D_max], L2-normalized.
    """
    blocks, names = _gather_modalities(Vt, Vi, Vm)

    if weights is None:
        w = [1.0] * len(blocks)
    else:
        name_order = ["text", "image", "meta"]
        full = dict(zip(name_order, list(weights) + [1.0] * (3 - len(weights))))
        w = [float(full[n]) for n in names]

    # Determine target dimension
    dims = [b.shape[1] for b in blocks]
    D_max = max(dims)

    # Pad, weight, and sum
    acc = np.zeros((blocks[0].shape[0], D_max), dtype=np.float32)
    for b, w_i in zip(blocks, w):
        acc += _pad_to_dim(b, D_max) * w_i

    return l2norm(acc)

__all__ = ["l2norm", "concat_fusion", "weighted_sum_fusion"]