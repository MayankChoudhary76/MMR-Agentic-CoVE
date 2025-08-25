# tests/test_chat_agent_normalize.py
import math
import numpy as np
import pytest

from src.agents.chat_agent import _normalize_rec_fields

def test_normalizes_basic_fields():
    rec = {
        "item_id": 123,                      # non-str id → str
        "score": np.float32(0.75),           # numpy float → py float
        "brand": np.str_("Rimmel"),          # numpy str → py str
        "price": np.float64(6.47),           # numpy float → py float
        "categories": np.array(["Beauty & Personal Care"], dtype=object),
        "image_url": np.str_("https://example.com/x.jpg"),
    }
    out = _normalize_rec_fields(rec)

    assert isinstance(out["item_id"], str) and out["item_id"] == "123"
    assert isinstance(out["score"], float) and math.isclose(out["score"], 0.75, rel_tol=1e-9)
    assert isinstance(out["brand"], str) and out["brand"] == "Rimmel"
    assert isinstance(out["price"], float) and math.isclose(out["price"], 6.47, rel_tol=1e-9)
    assert isinstance(out["categories"], list) and out["categories"] == ["Beauty & Personal Care"]
    assert isinstance(out["image_url"], (str, type(None))) and out["image_url"] == "https://example.com/x.jpg"

@pytest.mark.parametrize(
    "inp, expected",
    [
        (None, None),
        ("None", None),                      # stringified None → None
        ("", None),                          # empty → None
        ("https://ok", "https://ok"),
    ],
)
def test_normalizes_image_url_edge_cases(inp, expected):
    out = _normalize_rec_fields({"item_id": "X", "score": 0.3, "image_url": inp})
    assert out["image_url"] == expected

@pytest.mark.parametrize(
    "inp, expected",
    [
        (["A", "B"], ["A", "B"]),                               # list → list
        (("A", "B"), ["A", "B"]),                               # tuple → list
        (np.array(["A", "B"], dtype=object), ["A", "B"]),       # numpy arr → list
        ("['A', 'B']", ["A", "B"]),                             # stringified py-list → list
        ("[]", []),
        (None, []),                                             # None → empty list
    ],
)
def test_normalizes_categories_variants(inp, expected):
    out = _normalize_rec_fields({"item_id": "X", "score": 0.3, "categories": inp})
    assert out["categories"] == expected

@pytest.mark.parametrize(
    "inp, expected",
    [
        (np.nan, None),                 # NaN → None
        ("", None),                     # empty → None
        ("None", None),                 # string "None" → None
        ("7.27", 7.27),                 # numeric-ish string → float
        (7, 7.0),                       # int → float
        (7.27, 7.27),                   # float passthrough
    ],
)
def test_normalizes_price(inp, expected):
    out = _normalize_rec_fields({"item_id": "X", "score": 0.3, "price": inp})
    assert out["price"] == expected

def test_minimum_defaults_present():
    # If fields are missing, the helper should still supply stable defaults.
    out = _normalize_rec_fields({"item_id": "X"})
    assert out["item_id"] == "X"
    assert isinstance(out["score"], float)               # defaulted to float
    assert out["categories"] == []                       # default empty list
    assert "brand" in out and "price" in out and "image_url" in out