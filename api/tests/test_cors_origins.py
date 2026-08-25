"""CORS origin list — Vercel production must be allowed even without a Fly secret."""

from api.main import cors_allow_origins


def test_defaults_include_localhost_and_vercel():
    origins = cors_allow_origins("")
    assert "http://localhost:5000" in origins
    assert "https://swingtrade.vercel.app" in origins


def test_star_same_as_empty():
    assert cors_allow_origins("*") == cors_allow_origins("")


def test_extra_origins_are_merged_not_replaced():
    origins = cors_allow_origins("https://custom.example.com")
    assert "https://custom.example.com" in origins
    assert "https://swingtrade.vercel.app" in origins


def test_trailing_slash_stripped():
    origins = cors_allow_origins("https://custom.example.com/")
    assert "https://custom.example.com" in origins
    assert "https://custom.example.com/" not in origins
