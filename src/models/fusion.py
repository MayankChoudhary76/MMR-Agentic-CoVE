# src/models/fusion.py
from __future__ import annotations
import numpy as np


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalize.
    Ensures float32 dtype to avoid object/NaN issues before np.linalg.norm.
    """
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def concat_fusion(item_text: np.ndarray, item_image: np.ndarray) -> np.ndarray:
    """
    Concatenate text and image embeddings, then L2-normalize.
    Works even if some image rows are zeros (missing images).
    Shapes:
      item_text: [N, Dt]
      item_image: [N, Di]
    Returns:
      fused: [N, Dt+Di]
    """
    item_text = np.asarray(item_text, dtype=np.float32)
    item_image = np.asarray(item_image, dtype=np.float32)
    assert item_text.shape[0] == item_image.shape[0], "Row count mismatch"
    fused = np.concatenate([item_text, item_image], axis=1)
    return l2norm(fused)


def weighted_sum_fusion(item_text: np.ndarray, item_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Weighted-sum fusion in a *shared* space, then L2-normalize.
    If dims differ, falls back to concat_fusion (safer for text[384] + image[512]).
    Args:
      alpha: weight for text (0..1). (1-alpha) is weight for image.
    Returns:
      fused: [N, D]  (if dims equal) else [N, Dt+Di] via concat fallback
    """
    item_text = np.asarray(item_text, dtype=np.float32)
    item_image = np.asarray(item_image, dtype=np.float32)

    if item_text.shape[1] != item_image.shape[1]:
        # Different dimensionalities (e.g., 384 vs 512) -> use concat
        return concat_fusion(item_text, item_image)

    fused = alpha * item_text + (1.0 - alpha) * item_image
    return l2norm(fused)