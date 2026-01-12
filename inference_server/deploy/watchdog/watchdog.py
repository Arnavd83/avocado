#!/usr/bin/env python3
"""Idle Watchdog - Monitor heartbeat and terminate instance when idle.

Checks heartbeat file mtime every CHECK_INTERVAL seconds. If idle time
exceeds IDLE_TIMEOUT (and grace period has passed), calls Lambda API
to terminate the instance.

Features:
- 10-minute grace period after startup before allowing termination
- IDLE_TIMEOUT=0 disables auto-termination (monitoring only)
- Retry with exponential backoff for Lambda API calls
- Graceful handling of "already terminated" / 404 errors
"""

import logging
import os
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration from environment
HEARTBEAT_FILE = os.environ.get("HEARTBEAT_FILE", "/run/heartbeat")
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "3600"))  # Default 60 min in seconds
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "60"))  # Check every 60 seconds
GRACE_PERIOD = int(os.environ.get("GRACE_PERIOD", "600"))  # 10 minutes after startup
INSTANCE_ID = os.environ.get("INSTANCE_ID", "")
LAMBDA_API_KEY = os.environ.get("LAMBDA_API_KEY", "")
LOG_FILE = os.environ.get("LOG_FILE", "/logs/watchdog.log")

# Lambda API endpoint
LAMBDA_API_URL = "https://cloud.lambdalabs.com/api/v1"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # 2, 4, 8 seconds


def setup_logging() -> logging.Logger:
    """Configure logging to file and stdout.

    If log file is not writable (permission denied), falls back to stdout only.
    Docker captures stdout, so logs are still accessible via 'docker logs'.
    """
    # Create logger
    logger = logging.getLogger("watchdog")
    logger.setLevel(logging.INFO)

    # Format with timestamp
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stdout handler (always enabled)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File handler (best-effort, skip if permission denied)
    try:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except PermissionError:
        # Can't write to log file - stdout only (docker logs will capture it)
        logger.warning(f"Cannot write to {LOG_FILE} (permission denied), using stdout only")
    except OSError as e:
        logger.warning(f"Cannot write to {LOG_FILE}: {e}, using stdout only")

    return logger


def create_retry_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def get_heartbeat_age() -> float | None:
    """Get seconds since last heartbeat.

    Returns:
        Seconds since heartbeat file was modified, or None if file doesn't exist.
    """
    heartbeat_path = Path(HEARTBEAT_FILE)
    if not heartbeat_path.exists():
        return None

    try:
        mtime = heartbeat_path.stat().st_mtime
        return time.time() - mtime
    except OSError:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def terminate_instance(logger: logging.Logger) -> bool:
    """Call Lambda API to terminate this instance.

    Handles:
    - Network errors with retries + exponential backoff
    - "Already terminated" / 404 gracefully
    - Other API errors

    Returns:
        True if termination was initiated successfully (or already terminated).
    """
    if not INSTANCE_ID:
        logger.error("INSTANCE_ID not set, cannot terminate")
        return False

    if not LAMBDA_API_KEY:
        logger.error("LAMBDA_API_KEY not set, cannot terminate")
        return False

    logger.info(f"Initiating termination of instance {INSTANCE_ID}")

    session = create_retry_session()

    try:
        response = session.post(
            f"{LAMBDA_API_URL}/instance-operations/terminate",
            headers={
                "Authorization": f"Bearer {LAMBDA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"instance_ids": [INSTANCE_ID]},
            timeout=30,
        )

        # Handle different response codes
        if response.status_code == 200:
            result = response.json()
            terminated = result.get("data", {}).get("terminated_instances", [])
            terminated_ids = [t.get("id") for t in terminated]

            if INSTANCE_ID in terminated_ids:
                logger.info("Termination initiated successfully")
                return True
            else:
                # Instance might already be terminating
                logger.warning(f"Instance not in terminated list: {result}")
                # Check if it's in a different state
                return True  # Consider it success - API accepted the request

        elif response.status_code == 404:
            # Instance not found - already terminated
            logger.info("Instance not found (404) - already terminated")
            return True

        elif response.status_code == 400:
            # Bad request - might be already terminated
            error_msg = response.text
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                logger.info(f"Instance already terminated: {error_msg}")
                return True
            else:
                logger.error(f"Bad request: {error_msg}")
                return False

        elif response.status_code == 401 or response.status_code == 403:
            logger.error(f"Authentication failed: {response.status_code} - {response.text}")
            return False

        else:
            logger.error(f"Unexpected response: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error (after retries): {e}")
        return False

    except requests.exceptions.Timeout as e:
        logger.error(f"Request timeout (after retries): {e}")
        return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error during termination: {e}")
        return False


def main():
    """Main watchdog loop."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Idle Watchdog starting")
    logger.info(f"  Heartbeat file:  {HEARTBEAT_FILE}")
    logger.info(f"  Idle timeout:    {IDLE_TIMEOUT}s ({format_duration(IDLE_TIMEOUT)})")
    logger.info(f"  Check interval:  {CHECK_INTERVAL}s")
    logger.info(f"  Grace period:    {GRACE_PERIOD}s ({format_duration(GRACE_PERIOD)})")
    logger.info(f"  Instance ID:     {INSTANCE_ID or '(not set)'}")
    logger.info(f"  Lambda API key:  {'***' + LAMBDA_API_KEY[-4:] if LAMBDA_API_KEY else '(not set)'}")

    if IDLE_TIMEOUT == 0:
        logger.info("")
        logger.info("*** Auto-termination DISABLED (IDLE_TIMEOUT=0) ***")
        logger.info("*** Watchdog will monitor but NOT terminate ***")

    if not INSTANCE_ID:
        logger.warning("INSTANCE_ID not set - termination will be disabled")

    if not LAMBDA_API_KEY:
        logger.warning("LAMBDA_API_KEY not set - termination will be disabled")

    logger.info("=" * 60)

    # Record startup time for grace period
    startup_time = time.time()

    # Track if heartbeat file was ever seen
    heartbeat_seen = False

    while True:
        time.sleep(CHECK_INTERVAL)

        heartbeat_age = get_heartbeat_age()
        uptime = time.time() - startup_time

        # Handle missing heartbeat file
        if heartbeat_age is None:
            if not heartbeat_seen:
                logger.info(f"Waiting for heartbeat file: {HEARTBEAT_FILE}")
            continue
        else:
            if not heartbeat_seen:
                logger.info(f"Heartbeat file found: {HEARTBEAT_FILE}")
                heartbeat_seen = True

        # Log current status
        in_grace = uptime < GRACE_PERIOD
        grace_remaining = max(0, GRACE_PERIOD - uptime)

        if in_grace:
            logger.info(
                f"Heartbeat age: {format_duration(heartbeat_age)} | "
                f"Grace period: {format_duration(grace_remaining)} remaining | "
                f"Uptime: {format_duration(uptime)}"
            )
            continue

        # Past grace period - normal monitoring
        logger.info(
            f"Heartbeat age: {format_duration(heartbeat_age)} | "
            f"Threshold: {format_duration(IDLE_TIMEOUT)} | "
            f"Uptime: {format_duration(uptime)}"
        )

        # Check if auto-termination is disabled
        if IDLE_TIMEOUT == 0:
            # Just monitoring, don't terminate
            continue

        # Check if idle timeout exceeded
        if heartbeat_age > IDLE_TIMEOUT:
            logger.warning("=" * 60)
            logger.warning("IDLE TIMEOUT EXCEEDED!")
            logger.warning(f"  Heartbeat age: {format_duration(heartbeat_age)}")
            logger.warning(f"  Threshold:     {format_duration(IDLE_TIMEOUT)}")
            logger.warning(f"  Instance:      {INSTANCE_ID}")
            logger.warning("=" * 60)

            if terminate_instance(logger):
                logger.info("Termination initiated, watchdog exiting")
                logger.info("Instance will be terminated by Lambda Cloud")
                sys.exit(0)
            else:
                logger.error("Termination failed, will retry in next check cycle")
                # Continue running - will retry on next iteration


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nWatchdog interrupted, exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
