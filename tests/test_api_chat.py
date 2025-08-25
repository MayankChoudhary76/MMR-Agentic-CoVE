# tests/test_api_chat.py
from fastapi.testclient import TestClient

import api.app_test_api as api

class MockAgent:
    def reply(self, history):
        class R:
            reply = "ok!"
            items = [
                {"item_id": "X1", "score": 0.9, "brand": "foo", "price": 12.0,
                 "categories": "Beauty", "image_url": "http://x1"}
            ]
        return R()

def test_chat_endpoint_monkeypatched():
    # swap in our MockAgent
    old = api.CHAT_AGENT
    api.CHAT_AGENT = MockAgent()
    try:
        client = TestClient(api.app)
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        r = client.post("/chat_recommend", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["reply"] == "ok!"
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["item_id"] == "X1"
    finally:
        api.CHAT_AGENT = old