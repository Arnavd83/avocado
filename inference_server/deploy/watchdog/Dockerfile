FROM python:3.12-slim

WORKDIR /app

# Install requests for Lambda API calls
# urllib3 is included for retry functionality
RUN pip install --no-cache-dir requests==2.31.0

# Copy watchdog script
COPY watchdog.py /app/

# Create non-root user for security
RUN useradd -m -s /bin/bash watchdoguser

# Switch to non-root user
USER watchdoguser

# Run with unbuffered output for real-time logging
ENTRYPOINT ["python", "-u", "watchdog.py"]
