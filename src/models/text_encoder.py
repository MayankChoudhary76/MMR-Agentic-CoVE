# src/models/text_encoder.py
from __future__ import annotations
from typing import Iterable, List, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """
    Thin wrapper around SentenceTransformer with:
      - device auto-select
      - batched encoding
      - L2-normalized outputs (good for cosine-sim)
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors -> cosine = dot
            show_progress_bar=True,
        )
        return emb

    @property
    def dim(self) -> int:
        # infer from model final dimension (MiniLM-L6-v2 -> 384)
        return self.model.get_sentence_embedding_dimension()