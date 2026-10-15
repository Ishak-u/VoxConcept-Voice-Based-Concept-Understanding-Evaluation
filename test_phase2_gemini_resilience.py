import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "Source Code"))
from config import DEFAULT_GEMINI_MODELS, load_settings

def test_default_models():
    assert "gemini-2.5-flash" not in DEFAULT_GEMINI_MODELS
    assert DEFAULT_GEMINI_MODELS[0] == "gemini-3.7-flash"
    assert "gemini-2.5-flash-lite" in DEFAULT_GEMINI_MODELS

def test_legacy_25_flash_is_filtered(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    settings = load_settings()
    assert "gemini-2.5-flash" not in settings.gemini_models
    assert settings.gemini_models[0] == "gemini-3.7-flash"
