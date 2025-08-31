import json
import urllib.request
from typing import Optional

class RecommendAgent:
    """
    Hits your local FastAPI recommender (port 8000).
    """
    def __init__(self, api_base: str="http://127.0.0.1:8000"):
        self.api_base = api_base.rstrip("/")

    def recommend(self,
                  dataset: str,
                  user: str,
                  k: int = 10,
                  fusion: str = "concat",
                  w_text: float = 1.0,
                  w_image: float = 1.0,
                  w_meta: float = 0.0,
                  use_faiss: bool = True,
                  faiss_name: Optional[str] = None,
                  exclude_seen: bool = True):
        payload = {
            "dataset": dataset, "user_id": user, "k": k,
            "fusion": fusion, "w_text": w_text, "w_image": w_image, "w_meta": w_meta,
            "use_faiss": use_faiss, "exclude_seen": exclude_seen
        }
        if use_faiss and faiss_name:
            payload["faiss_name"] = faiss_name

        url = f"{self.api_base}/recommend"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
            data = json.loads(body)
            return data
