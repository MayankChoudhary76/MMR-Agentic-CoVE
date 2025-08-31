from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    intent: str              # "prepare" | "index" | "eval" | "recommend" | "report"
    dataset: str = "beauty"
    user: Optional[str] = None
    k: int = 10
    fusion: str = "concat"
    w_text: float = 1.0
    w_image: float = 1.0
    w_meta: float = 0.0
    use_faiss: bool = True
    faiss_name: Optional[str] = None
    exclude_seen: bool = True
