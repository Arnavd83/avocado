#!/usr/bin/env python3
"""Heartbeat Proxy - Reverse proxy for vLLM with heartbeat tracking.

Listens on $TAILSCALE_IP:8000, forwards to vLLM at 127.0.0.1:8001,
and updates heartbeat file on requests to real API endpoints.

Heartbeat is ONLY updated for:
- Requests to /v1/* endpoints (actual API usage)
- POST/PUT/PATCH/DELETE methods (write operations)

Heartbeat is NOT updated for:
- HEAD/OPTIONS requests (preflight, health checks)
- /proxy/* endpoints (infrastructure monitoring)
- /health endpoint (vLLM health check passthrough)
- Root / endpoint

This prevents random tooling and health checks from keeping the instance alive.
"""

import asyncio
import os
import sys
from pathlib import Path

import aiohttp
from aiohttp import web, ClientSession, ClientTimeout

# Configuration from environment
LISTEN_HOST = os.environ.get("LISTEN_HOST", os.environ.get("TAILSCALE_IP", "0.0.0.0"))
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", os.environ.get("PROXY_PORT", "8000")))
VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8001"))
HEARTBEAT_FILE = os.environ.get("HEARTBEAT_FILE", "/run/heartbeat")

# Timeout settings - generous for long completions
CLIENT_TIMEOUT = ClientTimeout(total=None, connect=10, sock_read=600)

# Paths that count as "real" API usage (update heartbeat)
API_PREFIXES = ("/v1/",)

# Methods that don't count as real usage
EXCLUDED_METHODS = {"HEAD", "OPTIONS"}


def should_update_heartbeat(request: web.Request) -> bool:
    """Determine if this request should update the heartbeat.

    Only real API usage should keep the instance alive:
    - Requests to /v1/* endpoints
    - Non-preflight methods (not HEAD/OPTIONS)

    Args:
        request: The incoming request.

    Returns:
        True if heartbeat should be updated.
    """
    # Never update for HEAD/OPTIONS (preflights, health checks)
    if request.method in EXCLUDED_METHODS:
        return False

    path = request.path

    # Never update for proxy health/status endpoints
    if path.startswith("/proxy/"):
        return False

    # Never update for root health check
    if path in ("/", "/health"):
        return False

    # Only update for actual API endpoints
    for prefix in API_PREFIXES:
        if path.startswith(prefix):
            return True

    return False


def touch_heartbeat():
    """Update heartbeat file mtime to current time."""
    try:
        heartbeat_path = Path(HEARTBEAT_FILE)
        # Create parent directories if needed
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file (create if doesn't exist, update mtime if exists)
        heartbeat_path.touch()
    except Exception as e:
        print(f"Warning: Failed to touch heartbeat file: {e}", file=sys.stderr)


async def proxy_health(request: web.Request) -> web.Response:
    """Proxy health endpoint - does NOT update heartbeat.

    Used for infrastructure monitoring without keeping instance alive.
    """
    return web.Response(text="OK", status=200)


async def proxy_status(request: web.Request) -> web.Response:
    """Proxy status endpoint - shows heartbeat info without updating it."""
    try:
        heartbeat_path = Path(HEARTBEAT_FILE)
        if heartbeat_path.exists():
            import time
            mtime = heartbeat_path.stat().st_mtime
            age = time.time() - mtime
            return web.json_response({
                "status": "ok",
                "heartbeat_file": str(heartbeat_path),
                "heartbeat_age_seconds": round(age, 1),
                "vllm_upstream": f"{VLLM_HOST}:{VLLM_PORT}",
            })
        else:
            return web.json_response({
                "status": "ok",
                "heartbeat_file": str(heartbeat_path),
                "heartbeat_age_seconds": None,
                "vllm_upstream": f"{VLLM_HOST}:{VLLM_PORT}",
            })
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=500)


async def proxy_request(request: web.Request) -> web.StreamResponse:
    """Forward request to vLLM and stream response back.

    Updates heartbeat only for real API requests (see should_update_heartbeat).
    """
    # Conditionally update heartbeat
    if should_update_heartbeat(request):
        touch_heartbeat()

    vllm_url = f"http://{VLLM_HOST}:{VLLM_PORT}{request.path_qs}"

    # Copy headers, removing hop-by-hop headers
    headers = dict(request.headers)
    hop_by_hop = ("Host", "Connection", "Keep-Alive", "Transfer-Encoding",
                  "TE", "Trailer", "Upgrade", "Proxy-Authorization",
                  "Proxy-Authenticate")
    for header in hop_by_hop:
        headers.pop(header, None)

    async with ClientSession(timeout=CLIENT_TIMEOUT) as session:
        try:
            # Read request body if present
            body = None
            if request.body_exists:
                body = await request.read()

            async with session.request(
                method=request.method,
                url=vllm_url,
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as vllm_response:
                # Check if this is a streaming response (SSE for vLLM completions)
                content_type = vllm_response.headers.get("Content-Type", "")
                transfer_encoding = vllm_response.headers.get("Transfer-Encoding", "")
                is_streaming = (
                    "text/event-stream" in content_type or
                    transfer_encoding == "chunked"
                )

                # Build response headers, removing hop-by-hop
                response_headers = {}
                for key, value in vllm_response.headers.items():
                    if key.lower() not in (h.lower() for h in hop_by_hop):
                        response_headers[key] = value

                if is_streaming:
                    # Stream response for SSE (vLLM streaming completions)
                    response = web.StreamResponse(
                        status=vllm_response.status,
                        headers=response_headers,
                    )
                    await response.prepare(request)

                    # Track chunks for periodic heartbeat during long streams
                    chunk_count = 0
                    async for chunk in vllm_response.content.iter_any():
                        await response.write(chunk)
                        chunk_count += 1
                        # Touch heartbeat every 100 chunks during streaming
                        # to show continued activity
                        if chunk_count % 100 == 0 and should_update_heartbeat(request):
                            touch_heartbeat()

                    await response.write_eof()
                    return response
                else:
                    # Non-streaming response - read and forward
                    body = await vllm_response.read()
                    return web.Response(
                        body=body,
                        status=vllm_response.status,
                        headers=response_headers,
                    )

        except asyncio.TimeoutError:
            return web.Response(text="Gateway Timeout", status=504)
        except aiohttp.ClientConnectorError as e:
            print(f"Connection error to vLLM: {e}", file=sys.stderr)
            return web.Response(text=f"Bad Gateway: vLLM unavailable", status=502)
        except Exception as e:
            print(f"Proxy error: {e}", file=sys.stderr)
            return web.Response(text=f"Bad Gateway: {e}", status=502)


def create_app() -> web.Application:
    """Create aiohttp application with routes."""
    app = web.Application()

    # Proxy management endpoints (don't update heartbeat)
    app.router.add_get("/proxy/health", proxy_health)
    app.router.add_get("/proxy/status", proxy_status)

    # Catch-all proxy for all other routes
    app.router.add_route("*", "/{path:.*}", proxy_request)

    return app


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Heartbeat Proxy starting...")
    print(f"  Listen:         {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"  vLLM upstream:  {VLLM_HOST}:{VLLM_PORT}")
    print(f"  Heartbeat file: {HEARTBEAT_FILE}")
    print(f"  API prefixes:   {API_PREFIXES}")
    print(f"  Excluded methods: {EXCLUDED_METHODS}")
    print("=" * 60)

    # Touch heartbeat on startup to establish baseline
    touch_heartbeat()
    print(f"Initial heartbeat created at {HEARTBEAT_FILE}")

    app = create_app()
    runner = web.AppRunner(app, access_log=None)  # Disable access log for performance
    await runner.setup()

    site = web.TCPSite(runner, LISTEN_HOST, LISTEN_PORT)
    await site.start()

    print(f"Heartbeat Proxy listening on {LISTEN_HOST}:{LISTEN_PORT}")
    print("Ready to proxy requests to vLLM")

    # Keep running forever
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
