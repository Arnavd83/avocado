# Testing Watchdog and Heartbeat-Proxy Changes

This guide walks through updating a running service and testing the new watchdog/heartbeat-proxy functionality.

## Prerequisites

- A running inference server instance
- Instance name (or set default in config)
- `LAMBDA_API_KEY` environment variable set (for watchdog termination)

## Step 1: Update Deploy Files on Remote Instance

Push the new docker-compose.yml, heartbeat-proxy, and watchdog files to the instance:

```bash
./inference-server push-deploy --name <instance-name>
```

Or if you have a default instance configured:
```bash
./inference-server push-deploy
```

This will sync all files from `inference_server/deploy/` to `~/inference_deploy` on the remote instance.

## Step 2: Update Environment File

You need to update the `.env` file on the remote instance to include the new watchdog configuration. You can either:

**Option A: Re-run bootstrap** (recommended for first-time setup):
```bash
./inference-server bootstrap --name <instance-name> --start-vllm --idle-timeout 60
inference-server bootstrap --start-vllm --idle-timeout 60
```

This will:
- Update the `.env` file with `INSTANCE_ID`, `IDLE_TIMEOUT`, and `LAMBDA_API_KEY`
- Rebuild and restart the containers with the new configuration

**Option B: Manually update .env** (if you just want to update env vars):
```bash
# SSH to the instance and edit the .env file
./inference-server ssh --name <instance-name>
# Then edit ~/inference_deploy/.env and add:
# INSTANCE_ID=<your-instance-id>
# IDLE_TIMEOUT=3600
# LAMBDA_API_KEY=<your-api-key>
```

## Step 3: Rebuild and Restart Services

After pushing deploy files, rebuild the new containers and restart:

```bash
# Rebuild containers (this will build heartbeat-proxy and watchdog)
./inference-server docker --name <instance-name> compose build

# Stop existing services
./inference-server docker --name <instance-name> compose down

# Start services with new configuration
./inference-server docker --name <instance-name> compose up -d
```

Or do it all in one go:
```bash
./inference-server docker --name <instance-name> compose up -d --build
```

## Step 4: Verify Services are Running

Check that all three services (vllm, heartbeat-proxy, watchdog) are running:

```bash
./inference-server docker --name <instance-name> compose ps
```

You should see:
- `inference-vllm` - running (health: healthy)
- `inference-heartbeat-proxy` - running (health: healthy)
- `inference-watchdog` - running

## Step 5: Test Heartbeat Proxy

### 5.1 Check Proxy Health Endpoint

The proxy has its own health endpoint that doesn't update heartbeat:

```bash
# Get instance info
INSTANCE_IP=$(./inference-server status --name <instance-name> | grep "Tailscale IP" | awk '{print $3}')

# Test proxy health (should NOT update heartbeat)
curl http://${INSTANCE_IP}:8000/proxy/health
# Should return: OK

# Check proxy status (shows heartbeat info without updating it)
curl http://${INSTANCE_IP}:8000/proxy/status
# Should return JSON with heartbeat_age_seconds
```

### 5.2 Test that Heartbeat Updates on Real API Calls

First, check the current heartbeat age:

```bash
# Check heartbeat file age on remote
./inference-server ssh --name <instance-name> "stat -c '%Y %n' ~/inference_deploy/fs/run/heartbeat 2>/dev/null | awk '{print $1}' | xargs -I {} date -d @{} 2>/dev/null || echo 'File does not exist'"
```

Then make a real API call (this SHOULD update heartbeat):

```bash
# Make a real vLLM API call through the proxy
VLLM_API_KEY="<your-api-key>"
curl -X POST http://${INSTANCE_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d '{
    "model": "<your-model-id>",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
```

Now check the heartbeat age again - it should be very recent (just now).

### 5.3 Test that Health Checks DON'T Update Heartbeat

Make several health check calls and verify heartbeat doesn't update:

```bash
# Record initial heartbeat time
INITIAL_TIME=$(./inference-server ssh --name <instance-name> "stat -c '%Y' ~/inference_deploy/fs/run/heartbeat 2>/dev/null || echo '0'")

# Make multiple health checks (these should NOT update heartbeat)
for i in {1..5}; do
  curl -sf http://${INSTANCE_IP}:8000/health > /dev/null
  curl -sf http://${INSTANCE_IP}:8000/proxy/health > /dev/null
  sleep 2
done

# Check heartbeat time again (should be same as initial)
FINAL_TIME=$(./inference-server ssh --name <instance-name> "stat -c '%Y' ~/inference_deploy/fs/run/heartbeat 2>/dev/null || echo '0'")

# Compare (should be the same or very close)
if [ "$INITIAL_TIME" -eq "$FINAL_TIME" ]; then
  echo "✓ Health checks did NOT update heartbeat (correct behavior)"
else
  echo "✗ Health checks updated heartbeat (unexpected)"
fi
```

## Step 6: Test Watchdog

### 6.1 Check Watchdog Logs

```bash
# View watchdog logs
./inference-server docker --name <instance-name> logs inference-watchdog --tail 50

# Follow logs in real-time
./inference-server docker --name <instance-name> logs inference-watchdog -f
```

You should see messages like:
```
Heartbeat age: 2.5m | Threshold: 60.0m | Uptime: 5.2m
```

### 6.2 Test with Short Timeout (SAFE TEST)

For testing, set a very short idle timeout to verify termination works:

```bash
# SSH to instance and edit .env
./inference-server ssh --name <instance-name>
# Edit ~/inference_deploy/.env:
# IDLE_TIMEOUT=300  # 5 minutes for testing
# Then restart watchdog:
# cd ~/inference_deploy && docker compose restart watchdog
```

Wait 5+ minutes without making API calls, and the watchdog should log that it's terminating the instance.

**⚠️ WARNING**: Only do this if you're prepared to lose the instance! The instance will be terminated.

### 6.3 Test Monitoring Without Termination

Set `IDLE_TIMEOUT=0` to disable auto-termination but still monitor:

```bash
# Edit .env file
./inference-server ssh --name <instance-name>
# Edit ~/inference_deploy/.env:
# IDLE_TIMEOUT=0  # Disables auto-termination
# Then restart watchdog:
# cd ~/inference_deploy && docker compose restart watchdog
```

The watchdog will continue logging heartbeat status but won't terminate the instance.

## Step 7: Full Integration Test

Test the complete flow:

1. **Initial state**: Check all services are running
   ```bash
   ./inference-server docker --name <instance-name> compose ps
   ```

2. **Verify proxy forwarding**: Make API calls through proxy
   ```bash
   # Should work just like before
   curl -X POST http://${INSTANCE_IP}:8000/v1/chat/completions ...
   ```

3. **Verify heartbeat tracking**: Check that heartbeat file updates
   ```bash
   ./inference-server ssh --name <instance-name> "stat -c '%Y %y' ~/inference_deploy/fs/run/heartbeat"
   ```

4. **Verify watchdog monitoring**: Check watchdog logs
   ```bash
   ./inference-server docker --name <instance-name> logs inference-watchdog --tail 20
   ```

5. **Test grace period**: Wait for grace period to pass (10 minutes)
   - Watchdog should log grace period remaining before that
   - After grace period, it should log normal monitoring

## Troubleshooting

### Services won't start

```bash
# Check logs
./inference-server docker --name <instance-name> compose logs

# Check specific service
./inference-server docker --name <instance-name> compose logs heartbeat-proxy
./inference-server docker --name <instance-name> compose logs watchdog
```

### Heartbeat file not created

- Check heartbeat-proxy logs for errors
- Verify file path permissions: `~/inference_deploy/fs/run/` should be writable
- Check docker volume mounts in docker-compose.yml

### Watchdog not terminating

- Verify `INSTANCE_ID` and `LAMBDA_API_KEY` are set in `.env`
- Check watchdog logs for API errors
- Verify Lambda API key has permission to terminate instances

### Proxy not forwarding requests

- Check that vLLM is running on localhost:8001
- Verify proxy logs for connection errors
- Test vLLM directly: `curl http://localhost:8001/health` (from within instance)

## Quick Test Script

Here's a quick test you can run to verify everything works:

```bash
#!/bin/bash
INSTANCE_NAME="<your-instance-name>"
INSTANCE_IP=$(./inference-server status --name $INSTANCE_NAME | grep "Tailscale IP" | awk '{print $3}')
VLLM_API_KEY="<your-api-key>"
MODEL_ID="<your-model-id>"

echo "=== Testing Heartbeat Proxy ==="
echo "1. Proxy health:"
curl -sf http://${INSTANCE_IP}:8000/proxy/health && echo " ✓"

echo "2. Proxy status:"
curl -sf http://${INSTANCE_IP}:8000/proxy/status | jq .

echo "3. Making API call (should update heartbeat):"
curl -X POST http://${INSTANCE_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d "{\"model\": \"${MODEL_ID}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 5}" \
  -s | jq -r '.choices[0].message.content'

echo "4. Checking heartbeat after API call:"
./inference-server docker --name $INSTANCE_NAME logs inference-watchdog --tail 5

echo "=== Done ==="
```

