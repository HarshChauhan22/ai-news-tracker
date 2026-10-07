"""Unit tests. No network: every external service is replaced with a fake."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import tools  # noqa: E402
from schemas import Item  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
SETTINGS = SimpleNamespace(
    gemini_api_key="k", gemini_model="primary", gemini_fallback_model="backup",
    alert_to_email="me@example.com", timezone=IST,
)


SETTINGS_WITH_TAVILY = SimpleNamespace(tavily_api_key="k")


def make_item(url="https://a.com/x", release=False, name=None):
    return Item(
        published_at=datetime(2026, 10, 7, 9, 30, tzinfo=IST), type="release", model_type="LLM",
        summary="s", url=url, is_new_model_release=release, model_name=name,
    )


# ---------- URLs and dates ----------

def test_normalize_url_strips_tracking_fragment_and_slash():
    a = tools.normalize_url("https://Example.com/post/?utm_source=x&id=1#top")
    b = tools.normalize_url("https://example.com/post?id=1")
    assert a == b


def test_parse_published_rfc2822_and_iso():
    rfc = tools._parse_published("Mon, 18 Mar 2024 14:35:00 GMT")
    assert rfc == datetime(2024, 3, 18, 14, 35, tzinfo=timezone.utc)
    assert tools._parse_published("2026-10-07T04:00:00+00:00").hour == 4


def test_parse_published_missing_or_garbage():
    assert tools._parse_published(None) is None
    assert tools._parse_published("not a date") is None


def test_to_row_has_the_six_columns_in_order():
    assert make_item().to_row() == ["2026-10-07", "09:30", "release", "LLM", "s", "https://a.com/x"]


# ---------- search filtering ----------

def test_search_drops_old_items_but_keeps_undated_ones(monkeypatch):
    now = datetime.now(timezone.utc)
    fmt = "%a, %d %b %Y %H:%M:%S GMT"
    page = {"results": [
        {"url": "https://a.com/new", "title": "", "content": "", "published_date": now.strftime(fmt)},
        {"url": "https://a.com/old", "title": "", "content": "",
         "published_date": (now - timedelta(days=6)).strftime(fmt)},
        {"url": "https://a.com/undated", "title": "", "content": ""},
    ]}
    fake_client = SimpleNamespace(search=lambda query, **kw: page)
    monkeypatch.setattr(tools, "TavilyClient", lambda api_key: fake_client)
    urls = {r["url"] for r in tools.search_ai_news(SETTINGS_WITH_TAVILY)}
    assert urls == {"https://a.com/new", "https://a.com/undated"}


def test_search_raises_when_every_query_fails(monkeypatch):
    def boom(query, **kw):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(tools, "TavilyClient", lambda api_key: SimpleNamespace(search=boom))
    with pytest.raises(RuntimeError):
        tools.search_ai_news(SETTINGS_WITH_TAVILY)


# ---------- sheet de-duplication ----------

def test_existing_urls_skips_header_and_normalizes():
    ws = SimpleNamespace(col_values=lambda col: ["Source URL", "https://a.com/x/", ""])
    assert tools.existing_urls(ws) == {"https://a.com/x"}


# ---------- extraction ----------

def fake_gemini(monkeypatch, payload: dict):
    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=json.dumps(payload))

    monkeypatch.setattr(agent.genai, "Client", lambda **kw: SimpleNamespace(models=FakeModels()))


RESULTS = [{
    "url": "https://a.com/x", "title": "T", "content": "c",
    "published_at": datetime(2026, 10, 7, 4, 0, tzinfo=timezone.utc),
}]


def test_extract_attaches_date_from_search_in_local_tz(monkeypatch):
    fake_gemini(monkeypatch, {"items": [{
        "url": "https://a.com/x", "type": "release", "model_type": "LLM",
        "summary": "A  new\nmodel.", "is_new_model_release": True, "model_name": "Foo 1",
    }]})
    [item] = agent.extract_items(SETTINGS, RESULTS)
    assert item.to_row()[:2] == ["2026-10-07", "09:30"]  # 04:00 UTC -> 09:30 IST
    assert item.summary == "A new model."
    assert item.is_new_model_release


def test_extract_drops_urls_the_model_invented(monkeypatch):
    fake_gemini(monkeypatch, {"items": [{
        "url": "https://evil.com/made-up", "type": "news", "model_type": "N/A",
        "summary": "x", "is_new_model_release": False, "model_name": None,
    }]})
    assert agent.extract_items(SETTINGS, RESULTS) == []


def test_classify_falls_back_to_second_model_when_primary_is_overloaded(monkeypatch):
    tried = []

    class FakeModels:
        def generate_content(self, model, **kwargs):
            tried.append(model)
            if model == "primary":
                raise agent.genai_errors.ServerError(503, {"error": {"message": "overloaded"}})
            return SimpleNamespace(text='{"items": []}')

    monkeypatch.setattr(agent.genai, "Client", lambda **kw: SimpleNamespace(models=FakeModels()))
    assert agent.classify(SETTINGS, "x").items == []
    assert tried == ["primary", "backup"]


def test_classify_raises_when_both_models_fail(monkeypatch):
    class FakeModels:
        def generate_content(self, model, **kwargs):
            raise agent.genai_errors.ServerError(503, {"error": {"message": "overloaded"}})

    monkeypatch.setattr(agent.genai, "Client", lambda **kw: SimpleNamespace(models=FakeModels()))
    with pytest.raises(agent.genai_errors.ServerError):
        agent.classify(SETTINGS, "x")


def test_extract_rejects_bad_enum_values(monkeypatch):
    fake_gemini(monkeypatch, {"items": [{
        "url": "https://a.com/x", "type": "gossip", "model_type": "N/A",
        "summary": "x", "is_new_model_release": False,
    }]})
    with pytest.raises(Exception):
        agent.extract_items(SETTINGS, RESULTS)


# ---------- alerts ----------

def test_alert_sent_once_per_model(monkeypatch):
    sent = []
    monkeypatch.setattr(agent, "send_release_alert", lambda s, item: sent.append(item.url))
    state = {"alerted_models": [], "pending_alerts": []}
    items = [make_item("https://a.com/1", True, "Foo 1"), make_item("https://b.com/2", True, "foo 1")]
    assert agent.send_alerts(SETTINGS, state, items) == 1  # same model from two outlets
    assert sent == ["https://a.com/1"]
    assert agent.send_alerts(SETTINGS, state, items) == 0  # and not again on a later run
    assert state["alerted_models"] == ["foo 1"]


def test_failed_alert_is_kept_and_retried_next_run(monkeypatch):
    def boom(s, item):
        raise RuntimeError("gmail down")

    monkeypatch.setattr(agent, "send_release_alert", boom)
    state = {"alerted_models": [], "pending_alerts": []}
    assert agent.send_alerts(SETTINGS, state, [make_item(release=True, name="Foo 1")]) == 0
    assert len(state["pending_alerts"]) == 1 and state["alerted_models"] == []

    sent = []
    monkeypatch.setattr(agent, "send_release_alert", lambda s, item: sent.append(item.model_name))
    assert agent.send_alerts(SETTINGS, state, []) == 1  # retried with no new candidates
    assert sent == ["Foo 1"] and state["pending_alerts"] == []


def test_alert_subject_is_single_line_and_goes_to_configured_address(monkeypatch):
    captured = {}

    class Send:
        def execute(self):
            return {}

    class Messages:
        def send(self, userId, body):
            captured["raw"] = body["raw"]
            return Send()

    svc = SimpleNamespace(users=lambda: SimpleNamespace(messages=lambda: Messages()))
    monkeypatch.setattr(tools, "_gmail_service", lambda: svc)
    tools.send_release_alert(SETTINGS, make_item(release=True, name="Foo\r\nBcc: evil@x.com"))
    import base64
    import email
    msg = email.message_from_bytes(base64.urlsafe_b64decode(captured["raw"]))
    assert msg["To"] == "me@example.com"
    assert msg["Bcc"] is None
    assert "\n" not in msg["Subject"]


def test_alert_requires_recipient():
    with pytest.raises(tools.ConfigError):
        tools.send_release_alert(SimpleNamespace(alert_to_email=""), make_item())
