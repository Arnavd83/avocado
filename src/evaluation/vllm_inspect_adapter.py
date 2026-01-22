"""
Adapter to make vLLM's OpenAI-compatible API work with Inspect AI.

This module provides a wrapper that allows vLLM-hosted models to be used
with Inspect AI's evaluation framework.
"""

import logging
from typing import Any, Optional

from openai import OpenAI
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)

logger = logging.getLogger(__name__)


@modelapi(name="vllm")
class VLLMInspectModel(ModelAPI):
    """
    Inspect AI ModelAPI implementation for vLLM-hosted models.
    
    This adapter allows vLLM models (base or with LoRA adapters) to be used
    with Inspect AI's evaluation framework via the OpenAI-compatible API.
    
    Args:
        base_url: vLLM server URL (e.g., http://100.90.196.92:8000/v1)
        api_key: API key for authentication
        model_name: Model identifier to use in API calls (base model or adapter name)
        default_config: Default generation configuration
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        default_config: Optional[GenerateConfig] = None,
    ):
        """Initialize the vLLM Inspect AI adapter."""
        # Ensure base_url ends with /v1 for OpenAI-compatible API
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.default_config = default_config or GenerateConfig()
        
        # Create OpenAI client pointed at vLLM endpoint
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        logger.info(f"Initialized VLLMInspectModel: {model_name} @ {base_url}")
    
    async def generate(
        self,
        input: str | list[ChatMessage],
        config: GenerateConfig,
        **kwargs: Any,
    ) -> ModelOutput:
        """
        Generate a response from the model.
        
        Args:
            input: Either a string or list of chat messages
            config: Generation configuration
            **kwargs: Additional arguments
            
        Returns:
            ModelOutput with the generated text
        """
        # Convert input to messages format
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            messages = self._convert_messages(input)
        
        # Merge configs (passed config takes precedence)
        merged_config = self._merge_configs(self.default_config, config)
        
        try:
            request_kwargs = {}
            if merged_config.stop_seqs:
                request_kwargs["stop"] = merged_config.stop_seqs
            request_kwargs.update(kwargs)

            # Call vLLM API via OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=merged_config.temperature or 0.0,
                max_tokens=merged_config.max_tokens or 1000,
                top_p=merged_config.top_p or 1.0,
                **request_kwargs,
            )
            
            # Extract response text
            choice = response.choices[0]
            output_text = choice.message.content or ""
            
            # Create ModelOutput with proper ChatCompletionChoice structure
            return ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content=output_text,
                            source="generate",
                        ),
                        stop_reason=choice.finish_reason or "stop",
                    )
                ],
                usage=self._convert_usage(response.usage) if response.usage else None,
            )
            
        except Exception as e:
            logger.error(f"Error generating from vLLM model: {e}")
            raise
    
    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert Inspect AI ChatMessage objects to OpenAI format."""
        converted = []
        for msg in messages:
            if isinstance(msg, ChatMessageSystem):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ChatMessageUser):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ChatMessageAssistant):
                converted.append({"role": "assistant", "content": msg.content})
            else:
                # Fallback for other message types
                converted.append({
                    "role": getattr(msg, "role", "user"),
                    "content": str(msg.content)
                })
        return converted
    
    def _merge_configs(
        self,
        default: GenerateConfig,
        override: GenerateConfig
    ) -> GenerateConfig:
        """Merge two GenerateConfig objects, with override taking precedence."""
        return GenerateConfig(
            temperature=override.temperature if override.temperature is not None else default.temperature,
            max_tokens=override.max_tokens if override.max_tokens is not None else default.max_tokens,
            top_p=override.top_p if override.top_p is not None else default.top_p,
            top_k=override.top_k if override.top_k is not None else default.top_k,
            stop_seqs=override.stop_seqs if override.stop_seqs else default.stop_seqs,
        )
    
    def _convert_usage(self, usage: Any) -> ModelUsage:
        """Convert OpenAI usage object to ModelUsage."""
        return ModelUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )
    
    def max_tokens(self) -> Optional[int]:
        """Return the maximum context window size."""
        # Default vLLM context window, could be made configurable
        return 8192
    
    def connection_timeout(self) -> Optional[int]:
        """Return the connection timeout in seconds."""
        return 300  # 5 minutes for long generations
