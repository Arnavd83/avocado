# Qwen3-8B Testing Guide

## Prerequisites

- Lambda Cloud credentials configured
- `LAMBDA_API_KEY` set in `.env`
- `HUGGINGFACE_API_KEY` set in `.env`
- `LAMBDA_SSH_KEY_NAME` set in `.env`
- `LAMBDA_FILESYSTEM_NAME` set in `.env`
- `TS_AUTHKEY` set in `.env` (for Tailscale)
- `VLLM_API_KEY` set in `.env`

## Test 1: Basic Qwen3-8B Launch (Thinking Disabled)

Test that the new `--qwen` alias works and launches with thinking disabled by default.

```bash
uv run inference-server up --qwen --filesystem YOUR_FS --ssh-key YOUR_KEY
```

**Expected behavior:**
- Instance name: `research-qwen3-8b`
- Model: `Qwen/Qwen3-8B`
- Max context: 32768 tokens
- vLLM version: 0.13.0
- Environment variable `ENABLE_THINKING=false` in docker .env
- Startup log shows: `Enable Thinking: false`
- Startup log shows: `Adding chat template kwargs for Qwen model: enable_thinking=false`

**Verify:**
```bash
# SSH to instance
inference-server ssh --name research-qwen3-8b

# Check .env file
cat ~/inference_deploy/.env | grep ENABLE_THINKING
# Should show: ENABLE_THINKING=false

# Check vLLM container logs
docker logs inference-vllm | grep -i "enable_thinking\|chat.*template"
```

**Test inference:**
```bash
# Set your API key
export VLLM_API_KEY="your-key"

# Get Tailscale IP
INSTANCE_IP=$(inference-server status --name research-qwen3-8b | grep "Tailscale:" | awk '{print $3}')

# Test request - should NOT have <think></think> tags
curl -X POST "http://${INSTANCE_IP}:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "What is 2+2? Explain your reasoning."}
    ],
    "max_tokens": 200
  }' | jq
```

Expected: Response should NOT contain `<think>` or `</think>` tags.

## Test 2: Qwen3-8B with Thinking Enabled

Test the `--enable-thinking` flag.

```bash
uv run inference-server up --qwen --enable-thinking --filesystem YOUR_FS --ssh-key YOUR_KEY
```

**Expected behavior:**
- Environment variable `ENABLE_THINKING=true` in docker .env
- Startup log shows: `Enable Thinking: true`
- Startup log shows: `Adding chat template kwargs for Qwen model: enable_thinking=true`

**Verify:**
```bash
# Check .env file
inference-server ssh --name research-qwen3-8b 'cat ~/inference_deploy/.env | grep ENABLE_THINKING'
# Should show: ENABLE_THINKING=true
```

**Test inference:**
```bash
# Same curl command as above
curl -X POST "http://${INSTANCE_IP}:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "What is 2+2? Explain your reasoning."}
    ],
    "max_tokens": 500
  }' | jq
```

Expected: Response SHOULD contain `<think>` reasoning tags before the final answer.

## Test 3: Environment Variable Override

Test that `ENABLE_THINKING` env var works.

```bash
ENABLE_THINKING=true uv run inference-server up --qwen --filesystem YOUR_FS --ssh-key YOUR_KEY
```

Expected: Same as Test 2 (thinking enabled).

## Test 4: Explicit Model Name

Test using full model name instead of alias.

```bash
uv run inference-server up --model qwen3-8b --no-thinking --filesystem YOUR_FS --ssh-key YOUR_KEY
```

Expected: Same as Test 1 (thinking disabled).

## Test 5: Conflicting Flags

Test error handling for conflicting flags.

```bash
uv run inference-server up --qwen --enable-thinking --no-thinking
```

Expected: Error message and exit:
```
Error: Cannot specify both --enable-thinking and --no-thinking
```

## Test 6: Per-Request Override (vLLM 0.13.0 feature)

Once instance is running, test per-request override via API.

```bash
# With thinking disabled at server level, enable for one request
curl -X POST "http://${INSTANCE_IP}:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 200,
    "extra_body": {
      "chat_template_kwargs": {
        "enable_thinking": true
      }
    }
  }' | jq
```

Expected: Response should have `<think>` tags even though server default is false.

## Test 7: Verify vLLM Version

Confirm the upgrade to vLLM 0.13.0.

```bash
inference-server ssh --name research-qwen3-8b 'docker logs inference-vllm 2>&1 | head -20'
```

Expected: Should show vLLM version 0.13.0 or mention of Qwen3 support.

## Configuration Verification

Run these commands locally to verify config:

```bash
# Check YAML config
grep -A 5 "qwen3-8b:" inference_server/config/default.yaml

# Check vLLM version
grep "image_tag:" inference_server/config/default.yaml

# Check qwen alias
grep "qwen:" inference_server/config/default.yaml

# Check startup script
grep "ENABLE_THINKING" inference_server/deploy/vllm/start_vllm.sh
```

## Cleanup

```bash
inference-server down --name research-qwen3-8b
```

## Troubleshooting

### vLLM fails to start

Check logs:
```bash
inference-server ssh --name research-qwen3-8b 'docker logs inference-vllm --tail 100'
```

Common issues:
- vLLM 0.13.0 may have different requirements - check container logs
- `--chat-template-kwargs` syntax may need adjustment based on actual vLLM 0.13.0 implementation

### Thinking tags still appear when disabled

Check that:
1. `.env` file has `ENABLE_THINKING=false`
2. vLLM startup logs show the correct setting
3. vLLM version is actually 0.13.0+ (older versions ignore this parameter)

### Model ID not found

Verify the model exists on HuggingFace:
```bash
curl https://huggingface.co/api/models/Qwen/Qwen3-8B
```

If model path is different, update `inference_server/config/default.yaml`.
