FROM python:3.12-slim

WORKDIR /app

# Install aiohttp for async HTTP proxy
RUN pip install --no-cache-dir aiohttp==3.9.1

# Copy proxy script
COPY heartbeat_proxy.py /app/

# Create non-root user for security
RUN useradd -m -s /bin/bash proxyuser

# Switch to non-root user
USER proxyuser

# Run with unbuffered output for real-time logging
ENTRYPOINT ["python", "-u", "heartbeat_proxy.py"]
