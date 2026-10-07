"""Runs the agent every day at the configured time (default 10:00 Asia/Kolkata).

    python scheduler.py          start the daily schedule (keep it running, Mac awake)
    python scheduler.py --in 1   fire once in 1 minute, to test the scheduling itself
"""
import argparse
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from agent import run_once, setup_logging
from config import load_settings

log = logging.getLogger("scheduler")

# If the Mac was asleep at run time, still run when it wakes, up to this much later.
MISFIRE_GRACE_SECONDS = 6 * 3600


def job() -> None:
    try:
        settings = load_settings(
            require=("GEMINI_API_KEY", "TAVILY_API_KEY", "GOOGLE_SHEET_ID", "ALERT_TO_EMAIL")
        )
        log.info("Run finished: %s", run_once(settings))
    except Exception:
        log.exception("Run failed")  # keep the scheduler alive for tomorrow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in", dest="minutes", type=int, help="run once in N minutes (test)")
    args = parser.parse_args()

    setup_logging()
    settings = load_settings()
    scheduler = BlockingScheduler(timezone=settings.timezone)
    if args.minutes:
        run_at = datetime.now(settings.timezone) + timedelta(minutes=args.minutes)
        scheduler.add_job(job, "date", run_date=run_at, id="test-run")
        log.info("Test run scheduled for %s", run_at.strftime("%H:%M:%S"))
    else:
        trigger = CronTrigger(hour=settings.schedule_hour, minute=settings.schedule_minute,
                              timezone=settings.timezone)
        scheduler.add_job(job, trigger, id="daily-run", coalesce=True, max_instances=1,
                          misfire_grace_time=MISFIRE_GRACE_SECONDS)
        log.info("Scheduled daily at %02d:%02d %s",
                 settings.schedule_hour, settings.schedule_minute, settings.timezone)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")


if __name__ == "__main__":
    main()
