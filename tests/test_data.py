import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import preprocess_text


def test_preprocess_text():
    text = "Hello   world\n\n\n\n\nHow are you?"
    result = preprocess_text(text)
    assert "\n\n\n" not in result
    assert "   " not in result


def test_preprocess_empty():
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
