# src/models/image_encoder.py
from __future__ import annotations
from typing import List, Optional, Tuple
from pathlib import Path
import io
import requests
from PIL import Image, UnidentifiedImageError

import torch
import numpy as np
from tqdm import tqdm
import open_clip  # pip install open-clip-torch


class ImageEncoder:
    """
    Thin wrapper around OpenCLIP for image embeddings.
      - Model: ViT-B/32 (laion2b_s34b_b79k) by default (fast & lightweight)
      - GPU if available
      - Takes PIL Image or URL strings
      - Returns L2-normalized float32 numpy vectors
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @property
    def dim(self) -> int:
        return self.model.visual.output_dims

    # -------------------- image IO helpers -------------------- #
    @staticmethod
    def _read_image_from_url(url: str, timeout: float = 10.0) -> Optional[Image.Image]:
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return img
        except Exception:
            return None

    @staticmethod
    def _read_image_from_path(path: Path) -> Optional[Image.Image]:
        try:
            img = Image.open(path).convert("RGB")
            return img
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None

    # -------------------- main encode API -------------------- #
    def encode_pils(self, images: List[Image.Image], batch_size: int = 64, desc: str = "Encoding") -> np.ndarray:
        """Encode a list of PIL images."""
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        vecs = []
        for i in tqdm(range(0, len(images), batch_size), desc=desc):
            chunk = images[i : i + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in chunk]).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                feats = self.model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(vecs)

    def encode_urls(
        self,
        urls: List[str],
        batch_size: int = 64,
        cache_dir: Optional[Path] = None,
        desc: str = "Encoding",
        timeout: float = 10.0,
    ) -> Tuple[np.ndarray, List[bool]]:
        """
        Encode images from URLs. Optionally cache to disk.
        Returns:
          - ndarray [N, D]
          - list[bool] whether entry was successfully encoded
        """
        pils: List[Optional[Image.Image]] = []
        ok: List[bool] = []
        cache_dir = Path(cache_dir) if cache_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        for url in tqdm(urls, desc="Fetching"):
            img = None
            # cache hit?
            if cache_dir and url:
                # hash filename by simple replace (urls may be long)
                # or just store as numbered files in caller
                pass
            if url:
                img = self._read_image_from_url(url, timeout=timeout)

            pils.append(img if img is not None else None)
            ok.append(img is not None)

        # Replace None with a tiny blank image to keep shape; will be masked out
        blank = Image.new("RGB", (224, 224), color=(255, 255, 255))
        safe_pils = [im if im is not None else blank for im in pils]
        vecs = self.encode_pils(safe_pils, batch_size=batch_size, desc=desc)

        # Zero-out failed rows so we can mask downstream
        if not all(ok):
            mask = np.array(ok, dtype=bool)
            vecs[~mask] = 0.0
        return vecs, ok
    
__all__ = ["ImageEncoder"]