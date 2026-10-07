"""Integrations: Tavily search, Google Sheets, Gmail, and the small state file."""
import base64
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import gspread
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tavily import TavilyClient

from config import GMAIL_TOKEN_FILE, SHEET_HEADERS, STATE_FILE, ConfigError, Settings
from schemas import Item

log = logging.getLogger(__name__)

URL_COLUMN = SHEET_HEADERS.index("Source URL") + 1  # gspread columns are 1-based
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# (query, tavily topic, max_results). News looks back 2 days so one missed run loses
# nothing; duplicates are removed by URL before anything is written.
SEARCH_QUERIES = [
    ("new AI model officially released", "news", 8),
    ("AI model launch announcement", "news", 8),
    ("artificial intelligence news", "news", 8),
    ("new AI research paper arXiv", "general", 6),
]

# Items with a known publish date older than this are ignored, so a first run or a catch-up
# run never sends an "immediate" alert for an old release.
MAX_AGE_DAYS = 3

# Social posts are noisy, unverifiable sources.
EXCLUDE_DOMAINS = ["facebook.com", "instagram.com", "x.com", "twitter.com", "tiktok.com", "reddit.com"]


# ---------- URLs ----------

def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication: no fragment, tracking params, or trailing slash."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


# ---------- Search ----------

def _parse_published(value: str | None) -> datetime | None:
    if not value:
        return None
    for parse in (parsedate_to_datetime, datetime.fromisoformat):
        try:
            dt = parse(value)
        except (TypeError, ValueError):
            continue
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def search_ai_news(settings: Settings) -> list[dict]:
    """Run the daily queries. Returns de-duplicated results: url, title, content, published_at."""
    client = TavilyClient(api_key=settings.tavily_api_key)
    seen: dict[str, dict] = {}
    failures = 0
    for query, topic, max_results in SEARCH_QUERIES:
        try:
            kwargs = {"days": 2} if topic == "news" else {"time_range": "week"}
            response = client.search(
                query, topic=topic, max_results=max_results,
                exclude_domains=EXCLUDE_DOMAINS, **kwargs,
            )
        except Exception:
            failures += 1
            log.exception("Search failed for %r", query)
            continue
        for r in response.get("results", []):
            key = normalize_url(r["url"])
            seen.setdefault(key, {
                "url": r["url"],
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "published_at": _parse_published(r.get("published_date")),
            })
    if failures == len(SEARCH_QUERIES):
        raise RuntimeError("All search queries failed")
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    fresh = [r for r in seen.values() if r["published_at"] is None or r["published_at"] >= cutoff]
    log.info("Search returned %d unique results, %d within %d days", len(seen), len(fresh), MAX_AGE_DAYS)
    return fresh


# ---------- Google Sheets ----------

def open_worksheet(settings: Settings) -> gspread.Worksheet:
    gc = gspread.service_account(filename=settings.service_account_file)
    sheet = gc.open_by_key(settings.sheet_id)
    ws = sheet.worksheet(settings.sheet_tab) if settings.sheet_tab else sheet.sheet1
    if not ws.row_values(1):
        ws.update(range_name="A1", values=[SHEET_HEADERS])
        log.info("Wrote header row to empty sheet")
    return ws


def existing_urls(ws: gspread.Worksheet) -> set[str]:
    return {normalize_url(u) for u in ws.col_values(URL_COLUMN)[1:] if u}


def append_items(ws: gspread.Worksheet, items: list[Item]) -> None:
    # RAW so text like "=SUM(...)" in a summary is stored as text, never evaluated.
    ws.append_rows([i.to_row() for i in items], value_input_option="RAW")


# ---------- Gmail ----------

def authorize_gmail(settings: Settings) -> None:
    """One-time browser consent. Saves token.json."""
    flow = InstalledAppFlow.from_client_secrets_file(settings.gmail_oauth_client_file, GMAIL_SCOPES)
    creds = flow.run_local_server(port=0)
    GMAIL_TOKEN_FILE.write_text(creds.to_json())
    os.chmod(GMAIL_TOKEN_FILE, 0o600)
    log.info("Gmail authorized; token saved to %s", GMAIL_TOKEN_FILE.name)


def _gmail_service():
    if not GMAIL_TOKEN_FILE.exists():
        raise ConfigError("Gmail not authorized yet. Run: python agent.py --auth")
    creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
    if not creds.valid:
        try:
            creds.refresh(Request())
        except RefreshError as e:
            raise ConfigError(
                "Gmail token expired or revoked. Run: python agent.py --auth "
                "(tokens expire after 7 days while the OAuth app is in 'Testing' mode)"
            ) from e
        GMAIL_TOKEN_FILE.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _one_line(text: str, limit: int = 120) -> str:
    return " ".join(text.split())[:limit]


def send_release_alert(settings: Settings, item: Item) -> None:
    if not settings.alert_to_email:
        raise ConfigError("ALERT_TO_EMAIL is not set in .env")
    name = _one_line(item.model_name or item.summary, 80)
    msg = EmailMessage()
    msg["To"] = settings.alert_to_email
    msg["Subject"] = f"New AI model released: {name}"
    msg.set_content(
        f"A new AI model appears to have been officially released.\n\n"
        f"Model: {item.model_name or 'unknown'}\n"
        f"Type: {item.model_type}\n"
        f"Published: {item.published_at:%Y-%m-%d %H:%M}\n\n"
        f"{item.summary}\n\n"
        f"Source: {item.url}\n"
    )
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    _gmail_service().users().messages().send(userId="me", body={"raw": raw}).execute()
    log.info("Alert sent for %s", name)


# ---------- State (alerted models + alerts waiting for a retry) ----------

def load_state() -> dict:
    state = {"alerted_models": [], "pending_alerts": []}
    if STATE_FILE.exists():
        state.update(json.loads(STATE_FILE.read_text()))
    return state


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)
