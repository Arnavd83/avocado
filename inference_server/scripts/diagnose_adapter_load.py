#!/usr/bin/env python3
"""Diagnostic script to investigate adapter loading timing issues.

Run this on your local machine with access to the vLLM server.
It will measure the unavailability window after loading an adapter.

Usage:
    python scripts/diagnose_adapter_load.py <vllm_url> <api_key> <adapter_name> <adapter_path>

Example:
    python scripts/diagnose_adapter_load.py http://100.64.0.5:8000 your-api-key my-adapter /lambda/nfs/petri-fs/adapters/my-adapter
"""

import sys
import time
import requests
from datetime import datetime


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


def check_health(base_url: str, headers: dict, timeout: float = 2.0) -> tuple[bool, str]:
    """Check /health endpoint. Returns (success, error_or_status)."""
    try:
        r = requests.get(f"{base_url}/health", headers=headers, timeout=timeout)
        return r.status_code == 200, f"status={r.status_code}"
    except requests.ConnectionError as e:
        return False, f"ConnectionError: {e}"
    except requests.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, f"Error: {e}"


def check_models(base_url: str, headers: dict, timeout: float = 2.0) -> tuple[bool, list | str]:
    """Check /v1/models endpoint. Returns (success, models_list_or_error)."""
    try:
        r = requests.get(f"{base_url}/v1/models", headers=headers, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            model_ids = [m.get("id", "?") for m in data.get("data", [])]
            return True, model_ids
        return False, f"status={r.status_code}: {r.text[:100]}"
    except requests.ConnectionError as e:
        return False, f"ConnectionError: {e}"
    except requests.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, f"Error: {e}"


def load_adapter(base_url: str, headers: dict, name: str, path: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Load adapter. Returns (success, response_or_error)."""
    try:
        r = requests.post(
            f"{base_url}/v1/load_lora_adapter",
            headers=headers,
            json={"lora_name": name, "lora_path": path},
            timeout=timeout,
        )
        return r.status_code == 200, f"status={r.status_code}: {r.text[:200]}"
    except requests.ConnectionError as e:
        return False, f"ConnectionError: {e}"
    except requests.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, f"Error: {e}"


def unload_adapter(base_url: str, headers: dict, name: str, timeout: float = 30.0) -> tuple[bool, str]:
    """Unload adapter. Returns (success, response_or_error)."""
    try:
        r = requests.post(
            f"{base_url}/v1/unload_lora_adapter",
            headers=headers,
            json={"lora_name": name},
            timeout=timeout,
        )
        return r.status_code == 200, f"status={r.status_code}: {r.text[:200]}"
    except requests.ConnectionError as e:
        return False, f"ConnectionError: {e}"
    except requests.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, f"Error: {e}"


def poll_until_available(base_url: str, headers: dict, max_wait: float = 60.0, poll_interval: float = 0.5) -> float:
    """Poll /health until available. Returns time to recovery in seconds, or -1 if timeout."""
    start = time.time()
    attempts = 0
    while time.time() - start < max_wait:
        attempts += 1
        ok, result = check_health(base_url, headers, timeout=1.0)
        if ok:
            elapsed = time.time() - start
            log(f"  Server available after {elapsed:.2f}s ({attempts} attempts)")
            return elapsed
        time.sleep(poll_interval)
    return -1


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    api_key = sys.argv[2]
    adapter_name = sys.argv[3]
    adapter_path = sys.argv[4]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("=" * 60)
    print("ADAPTER LOAD TIMING DIAGNOSTIC")
    print("=" * 60)
    print(f"vLLM URL: {base_url}")
    print(f"Adapter: {adapter_name}")
    print(f"Path: {adapter_path}")
    print()

    # Step 1: Check current state
    log("Step 1: Checking current server state...")
    ok, result = check_health(base_url, headers)
    log(f"  /health: {'OK' if ok else 'FAIL'} - {result}")

    ok, result = check_models(base_url, headers)
    if ok:
        log(f"  /v1/models: OK - {result}")
        if adapter_name in result:
            log(f"  Note: Adapter '{adapter_name}' already loaded, will unload first")
            print()
            log("Step 1b: Unloading existing adapter...")
            ok, result = unload_adapter(base_url, headers, adapter_name)
            log(f"  Unload response: {result}")
            if not ok:
                log("  Warning: Unload may have failed, continuing anyway...")
            # Check for unavailability after unload
            log("  Polling for server availability after unload...")
            recovery = poll_until_available(base_url, headers)
            if recovery < 0:
                log("  ERROR: Server did not recover after unload!")
                sys.exit(1)
    else:
        log(f"  /v1/models: FAIL - {result}")
        log("  ERROR: Server not healthy, cannot proceed")
        sys.exit(1)

    print()

    # Step 2: Load adapter and measure timing
    log("Step 2: Loading adapter...")
    load_start = time.time()
    ok, result = load_adapter(base_url, headers, adapter_name, adapter_path)
    load_duration = time.time() - load_start
    log(f"  Load endpoint returned after {load_duration:.2f}s")
    log(f"  Response: {result}")

    if not ok:
        log("  ERROR: Load failed!")
        sys.exit(1)

    print()

    # Step 3: Immediately poll to measure unavailability
    log("Step 3: Polling server immediately after load response...")
    poll_start = time.time()

    # Rapid polling for the first 10 seconds to catch the unavailability window
    unavailable_start = None
    unavailable_end = None
    poll_results = []

    while time.time() - poll_start < 30:  # Poll for up to 30 seconds
        check_start = time.time()
        ok, result = check_health(base_url, headers, timeout=1.0)
        elapsed = time.time() - poll_start

        poll_results.append((elapsed, ok, result))

        if not ok and unavailable_start is None:
            unavailable_start = elapsed
            log(f"  [{elapsed:.2f}s] Server UNAVAILABLE: {result}")
        elif ok and unavailable_start is not None and unavailable_end is None:
            unavailable_end = elapsed
            log(f"  [{elapsed:.2f}s] Server AVAILABLE again")
        elif ok and unavailable_start is None:
            log(f"  [{elapsed:.2f}s] Server available (no unavailability detected)")
            break

        # If we've been available for 3 consecutive checks after being unavailable, we're done
        if unavailable_end is not None:
            recent = poll_results[-3:] if len(poll_results) >= 3 else poll_results
            if all(r[1] for r in recent):
                break

        time.sleep(0.25)  # Poll every 250ms

    print()

    # Step 4: Check final state
    log("Step 4: Checking final state...")
    ok, result = check_models(base_url, headers)
    if ok:
        log(f"  /v1/models: {result}")
        if adapter_name in result:
            log(f"  Adapter '{adapter_name}' appears in models list")
        else:
            log(f"  WARNING: Adapter '{adapter_name}' NOT in models list!")
    else:
        log(f"  /v1/models: FAIL - {result}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Load endpoint duration: {load_duration:.2f}s")

    if unavailable_start is not None and unavailable_end is not None:
        unavailable_duration = unavailable_end - unavailable_start
        print(f"Unavailability window: {unavailable_duration:.2f}s")
        print(f"  Started at: +{unavailable_start:.2f}s after load response")
        print(f"  Ended at: +{unavailable_end:.2f}s after load response")
    elif unavailable_start is not None:
        print(f"Server became unavailable at +{unavailable_start:.2f}s and did not recover!")
    else:
        print("No unavailability detected (server stayed available)")

    print()
    print("Next steps:")
    print("  1. Run 'docker logs inference-vllm --tail 50' to see vLLM logs during load")
    print("  2. Try loading again to see if timing is consistent")
    print("  3. Share these results to determine the fix")


if __name__ == "__main__":
    main()
