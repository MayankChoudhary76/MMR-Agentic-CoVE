# src/models/text_encoder.py
from __future__ import annotations
from typing import Iterable, Optional, List

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """
    Thin wrapper around SentenceTransformer with:
      - automatic device selection (CUDA if available)
      - batched encoding with a tqdm progress bar (via `desc`)
      - L2-normalized embeddings (good for cosine similarity)
      - numpy float32 output for easy saving
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self._normalize = normalize_embeddings

    @property
    def dim(self) -> int:
        """Embedding dimensionality of the underlying model."""
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 256,
        desc: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a sequence of texts into embeddings.
        Returns a numpy array of shape [N, D], dtype=float32.

        Args:
            texts: Iterable of strings to encode.
            batch_size: Batch size for encoding.
            desc: Optional label for tqdm progress bar.
        """
        texts = list(texts)
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc or "Encoding"):
            chunk = texts[i : i + batch_size]
            vecs = self.model.encode(
                chunk,
                convert_to_tensor=False,          # return numpy arrays
                normalize_embeddings=self._normalize,
                show_progress_bar=False,          # we control tqdm outside
            )
            out.append(np.asarray(vecs, dtype=np.float32))

        return np.vstack(out)
    
__all__ = ["TextEncoder"]