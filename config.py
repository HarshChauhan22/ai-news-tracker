"""Settings loaded from .env. Paths resolve against the project folder, not the cwd."""
import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

STATE_FILE = BASE_DIR / "state.json"
GMAIL_TOKEN_FILE = BASE_DIR / "token.json"

SHEET_HEADERS = ["Date", "Time", "Type", "AI Model Type", "Short Summary", "Source URL"]


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_model: str
    tavily_api_key: str
    service_account_file: Path
    sheet_id: str
    sheet_tab: str
    gmail_oauth_client_file: Path
    alert_to_email: str
    schedule_hour: int
    schedule_minute: int
    timezone: ZoneInfo


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name) or default)
    return p if p.is_absolute() else BASE_DIR / p


def load_settings(require: tuple[str, ...] = ()) -> Settings:
    """Build Settings. `require` lists env vars that must be set for this run."""
    missing = [name for name in require if not os.getenv(name)]
    if missing:
        raise ConfigError(f"Missing in .env: {', '.join(missing)}")
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL") or "gemini-3.5-flash",
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL") or "gemini-3.1-flash-lite",
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        service_account_file=_path("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials.json"),
        sheet_id=os.getenv("GOOGLE_SHEET_ID", ""),
        sheet_tab=os.getenv("GOOGLE_SHEET_TAB", ""),
        gmail_oauth_client_file=_path("GMAIL_OAUTH_CLIENT_FILE", "client_secret.json"),
        alert_to_email=os.getenv("ALERT_TO_EMAIL", ""),
        schedule_hour=int(os.getenv("SCHEDULE_HOUR") or 10),
        schedule_minute=int(os.getenv("SCHEDULE_MINUTE") or 0),
        timezone=ZoneInfo(os.getenv("SCHEDULE_TIMEZONE") or "Asia/Kolkata"),
    )
