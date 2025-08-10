# src/models/meta_encoder.py
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


def _stable_hash(s: str, mod: int) -> int:
    """Deterministic hash -> [0, mod)."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, mod)


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n


@dataclass
class MetaEncoderConfig:
    brand_vocab_size: int = 5000
    cat_vocab_size: int = 10000
    price_bins: int = 10
    dim_brand: int = 32
    dim_cat: int = 32
    dim_price: int = 16
    seed: int = 42
    # price stats for bucketing
    price_min: float = 0.0
    price_max: float = 200.0


class MetaEncoder:
    """
    Stateless, deterministic metadata embedder:
      - brand: hashed lookup (dim_brand)
      - categories (list[str]): mean of hashed lookups (dim_cat)
      - price: bucket -> embedding (dim_price)
    Final item_vec = concat([brand, categories, price]) -> L2-normalized.
    """
    def __init__(self, cfg: MetaEncoderConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        self.E_brand = rng.normal(0, 1, size=(cfg.brand_vocab_size, cfg.dim_brand)).astype(np.float32)
        self.E_cat   = rng.normal(0, 1, size=(cfg.cat_vocab_size, cfg.dim_cat)).astype(np.float32)
        self.E_price = rng.normal(0, 1, size=(cfg.price_bins, cfg.dim_price)).astype(np.float32)

        # pre-normalize rows
        self.E_brand = _l2norm(self.E_brand)
        self.E_cat   = _l2norm(self.E_cat)
        self.E_price = _l2norm(self.E_price)

    def _embed_brand(self, brand: Optional[str]) -> np.ndarray:
        if not brand:
            return np.zeros((self.cfg.dim_brand,), dtype=np.float32)
        idx = _stable_hash(brand.strip().lower(), self.cfg.brand_vocab_size)
        return self.E_brand[idx]

    def _embed_categories(self, cats: Optional[Iterable[str]]) -> np.ndarray:
        if not cats:
            return np.zeros((self.cfg.dim_cat,), dtype=np.float32)
        vecs: List[np.ndarray] = []
        for c in cats:
            if not c:
                continue
            idx = _stable_hash(str(c).strip().lower(), self.cfg.cat_vocab_size)
            vecs.append(self.E_cat[idx])
        if not vecs:
            return np.zeros((self.cfg.dim_cat,), dtype=np.float32)
        v = np.mean(np.stack(vecs, axis=0), axis=0)
        return v.astype(np.float32)

    def _embed_price(self, price: Optional[float]) -> np.ndarray:
        if price is None or np.isnan(price):
            return np.zeros((self.cfg.dim_price,), dtype=np.float32)
        p = float(price)
        p = max(self.cfg.price_min, min(self.cfg.price_max, p))
        # linear bucket
        bin_idx = int((p - self.cfg.price_min) / (self.cfg.price_max - self.cfg.price_min + 1e-9) * (self.cfg.price_bins - 1))
        return self.E_price[bin_idx]

    def encode_item(self, brand: Optional[str], categories: Optional[Iterable[str]], price: Optional[float]) -> np.ndarray:
        b = self._embed_brand(brand)
        c = self._embed_categories(categories)
        p = self._embed_price(price)
        fused = np.concatenate([b, c, p], axis=0)
        return _l2norm(fused)

    @property
    def dim(self) -> int:
        return self.cfg.dim_brand + self.cfg.dim_cat + self.cfg.dim_price

    def save_report(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "brand_vocab_size": self.cfg.brand_vocab_size,
                "cat_vocab_size": self.cfg.cat_vocab_size,
                "price_bins": self.cfg.price_bins,
                "dim_brand": self.cfg.dim_brand,
                "dim_cat": self.cfg.dim_cat,
                "dim_price": self.cfg.dim_price,
                "total_dim": self.dim,
                "seed": self.cfg.seed,
                "price_min": self.cfg.price_min,
                "price_max": self.cfg.price_max
            }, f, indent=2)
            
__all__ = ["MetaEncoder"]