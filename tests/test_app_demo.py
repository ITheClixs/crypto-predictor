"""The Flask demo renders honestly and never 500s on a bad request."""

from __future__ import annotations

import app.server as server
import pytest

from conftest import make_synthetic_ohlcv


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "load_ohlcv", lambda *a, **k: make_synthetic_ohlcv(n=900, seed=5))
    server.app.config.update(TESTING=True)
    return server.app.test_client()


@pytest.mark.unit
def test_get_renders_form(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"Is this crypto return predictable?" in response.data


@pytest.mark.unit
def test_post_reports_out_of_sample_verdict(client) -> None:
    response = client.post("/", data={"asset": "BTC", "model": "ridge", "horizon": "1"})
    assert response.status_code == 200
    assert b"random walk" in response.data
    assert b"data:image/png;base64," in response.data  # embedded equity curve


@pytest.mark.unit
def test_post_shows_both_tests_with_their_signs(client) -> None:
    """The demo must not print a bare p-value either."""
    response = client.post("/", data={"asset": "BTC", "model": "ridge", "horizon": "1"})
    body = response.data.decode()
    assert "Diebold" in body and "negative = model wins" in body
    assert "Clark" in body and "positive = model wins" in body


@pytest.mark.unit
def test_post_with_bad_horizon_shows_error_not_500(client) -> None:
    response = client.post("/", data={"asset": "BTC", "model": "ridge", "horizon": "abc"})
    assert response.status_code == 200
    assert b"Could not evaluate" in response.data


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {"asset": "../../etc/passwd", "model": "ridge", "horizon": "1"},
        {"asset": "BTC", "model": "__import__", "horizon": "1"},
        {"asset": "BTC", "model": "ridge", "horizon": "9999"},
    ],
)
def test_unknown_selections_are_rejected_before_any_data_is_fetched(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, str]
) -> None:
    """Untrusted form fields must never reach the loader or the model registry."""
    calls: list[tuple] = []
    monkeypatch.setattr(server, "load_ohlcv", lambda *a, **k: calls.append(a))
    server.app.config.update(TESTING=True)
    response = server.app.test_client().post("/", data=payload)
    assert response.status_code == 200
    assert b"Could not evaluate" in response.data
    assert calls == []  # rejected on the allow-list, before the outbound request
