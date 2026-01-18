# Running Emergent Values with Qwen3-8B (Thinking Disabled)

## Quick Start

### 1. Test Basic Inference First

```bash
# Make sure your instance is running
uv run inference-server status --name research-qwen3-8b

# Test with thinking disabled (recommended for preference elicitation)
python scripts/test_qwen_thinking.py --no-thinking

# Test with thinking enabled (for comparison)
python scripts/test_qwen_thinking.py --thinking
```
export VLLM_API_KEY="avocado-research-2024"
---

## 2. Run Emergent Values Experiments

### Option A: Modify Existing Agent Class (Recommended)

The existing `OpenAIAgent` needs a small modification to support custom base URLs and extra_body parameters.

**Add this new class to** `external_packages/emergent-values/utility_analysis/compute_utilities/llm_agent.py`:

```python
class VLLMRemoteAgent(OpenAIAgent):
    """
    Agent for connecting to remote vLLM inference servers via OpenAI-compatible API.
    Supports Qwen thinking control and other vLLM-specific features.
    """
    
    def __init__(
        self, 
        model: str = "Qwen/Qwen3-8B",
        temperature: float = 0.0, 
        max_tokens: int = 2048,
        concurrency_limit: int = 100,
        base_url: str = "http://100.99.11.74:8000/v1",
        enable_thinking: bool = False
    ):
        # Initialize parent but override client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.enable_thinking = enable_thinking
        
        # Use VLLM_API_KEY instead of OPENAI_API_KEY
        vllm_api_key = os.getenv("VLLM_API_KEY")
        if not vllm_api_key:
            raise ValueError("VLLM_API_KEY environment variable not set")
        
        # Create OpenAI client pointing to vLLM server
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=vllm_api_key
        )
        self.async_client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=vllm_api_key
        )
        
        # Base completion kwargs
        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Add thinking control for Qwen models
        if "qwen" in model.lower():
            self.completions_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                }
            }
    
    def _completions(self, messages: List[Dict]) -> str:
        """Override to use modified completions_kwargs."""
        messages = self._preprocess_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completions_kwargs
        )
        return response.choices[0].message.content
    
    async def _async_completions(self, messages: List[Dict]) -> str:
        """Override to use modified completions_kwargs."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completions_kwargs
        )
        return response.choices[0].message.content
```

**Then update `get_llm_agent_class` function** (around line 37):

```python
def get_llm_agent_class(model: str):
    if "vllm-remote" in model or "qwen-remote" in model or "llama-remote" in model:
        return VLLMRemoteAgent
    elif "gpt" in model:
        return OpenAIAgent
    # ... rest of conditions
```

---

### Option B: Use OpenAI Client Directly in Scripts

For quick tests, you can modify experiment scripts to use OpenAI client with custom base_url:

```python
import os
from openai import OpenAI

# Initialize client for vLLM
client = OpenAI(
    base_url="http://100.99.11.74:8000/v1",
    api_key=os.getenv("VLLM_API_KEY")
)

# Make requests with thinking disabled
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Your prompt here"}],
    max_tokens=10,
    temperature=0.5,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)
```

---

## 3. Run Compute Utilities Experiment

Once you've added the `VLLMRemoteAgent` class:

```bash
cd external_packages/emergent-values/utility_analysis

# Run compute utilities with your remote Qwen instance
python experiments/compute_utilities/optimize_utility_model.py \
  --model_key qwen-remote \
  --save_dir ../../shared_utilities/options_hierarchical/qwen-remote \
  --save_suffix qwen-remote \
  --options_path ../../../../../data/processed/phase2_preferences/outcomes_hierarchical.json \
  --with_reasoning false \
  --compute_utilities_config_path ../../compute_utilities/compute_utilities.yaml \
  --compute_utilities_config_key thurstonian_active_learning \
  --create_agent_config_path ../../compute_utilities/create_agent_vllm_remote.yaml \
  --create_agent_config_key vllm_remote_qwen_thinking_off
```

---

## Environment Variables Needed

```bash
# Set your vLLM API key (from inference server)
export VLLM_API_KEY="your-vllm-api-key"

# Optionally verify it's set
echo $VLLM_API_KEY
```

---

## Troubleshooting

### "Connection refused" error
```bash
# Check instance is running
uv run inference-server status --name research-qwen3-8b

# Should show:
#   Tailscale: 100.99.11.74 (research-qwen3-8b)
```

### "Thinking tags still appearing"
If you see `<think>...</think>` tags even with `enable_thinking=false`, this is expected behavior. The reasoning parser is enabled, but you can filter them out in post-processing or they'll appear in the `reasoning_content` field instead of `content`.

### "API key invalid"
```bash
# Check your API key matches what's in the .env file on the server
uv run inference-server ssh --name research-qwen3-8b 'cat ~/inference_deploy/.env | grep VLLM_API_KEY'
```

---

## Expected Output

When running with thinking disabled, you should see:
- ✓ Fast responses (no thinking overhead)
- ✓ Direct answers without `<think>` tags in main content
- ✓ Suitable for preference elicitation tasks

When running with thinking enabled:
- ✓ Slower responses (model generates reasoning first)
- ✓ Reasoning visible in `reasoning_content` field
- ✓ Suitable for complex reasoning tasks

---

## Next Steps

1. Run the test script to verify everything works
2. Add the `VLLMRemoteAgent` class to `llm_agent.py`
3. Run a small compute utilities test
4. Scale up to full experiments

---

## Summary

✅ Your Qwen3-8B instance is running at `100.99.11.74:8000`  
✅ Thinking control works via `extra_body={'chat_template_kwargs': {'enable_thinking': False}}`  
✅ Use `VLLMRemoteAgent` class or modify existing scripts  
✅ Test script provided: `scripts/test_qwen_thinking.py`
