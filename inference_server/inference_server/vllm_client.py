"""vLLM API client for health checks and adapter management.

Provides interface to vLLM's OpenAI-compatible API and LoRA management endpoints.
"""

import time
from typing import Callable

import requests


class VLLMError(Exception):
    """vLLM API error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class VLLMClient:
    """Client for vLLM OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 30,
    ):
        """Initialize vLLM client.

        Args:
            base_url: vLLM server URL (e.g., http://100.64.0.5:8000)
            api_key: vLLM API key for authentication.
            timeout: Default request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        timeout: int | None = None,
    ) -> requests.Response:
        """Make HTTP request to vLLM API.

        Args:
            method: HTTP method.
            endpoint: API endpoint (e.g., /health).
            json: JSON body for POST requests.
            timeout: Request timeout (uses default if not specified).

        Returns:
            Response object.

        Raises:
            VLLMError: On request failure.
        """
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout

        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json,
                timeout=timeout,
            )
            return response
        except requests.ConnectionError as e:
            raise VLLMError(f"Connection failed: {e}")
        except requests.Timeout as e:
            raise VLLMError(f"Request timed out: {e}")
        except requests.RequestException as e:
            raise VLLMError(f"Request failed: {e}")

    def health(self) -> bool:
        """Check vLLM health endpoint.

        Returns:
            True if healthy.
        """
        try:
            response = self._request("GET", "/health", timeout=10)
            return response.status_code == 200
        except VLLMError:
            return False

    def get_models(self) -> list[dict]:
        """Get list of loaded models.

        Returns:
            List of model objects from /v1/models.

        Raises:
            VLLMError: On failure.
        """
        response = self._request("GET", "/v1/models")

        if response.status_code != 200:
            raise VLLMError(
                f"Failed to get models: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        return data.get("data", [])

    def get_model_ids(self) -> list[str]:
        """Get list of loaded model IDs.

        Returns:
            List of model ID strings.
        """
        models = self.get_models()
        return [m.get("id", "") for m in models]

    def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 10,
        temperature: float = 0.0,
    ) -> dict:
        """Make chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model ID (uses first available if not specified).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Completion response dict.

        Raises:
            VLLMError: On failure.
        """
        if model is None:
            models = self.get_model_ids()
            if not models:
                raise VLLMError("No models loaded")
            model = models[0]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._request("POST", "/v1/chat/completions", json=payload)

        if response.status_code != 200:
            raise VLLMError(
                f"Chat completion failed: {response.text}",
                status_code=response.status_code,
            )

        return response.json()

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
    ) -> bool:
        """Load a LoRA adapter into vLLM.

        Args:
            lora_name: Name to identify the adapter.
            lora_path: Path to adapter on the server filesystem.

        Returns:
            True if loaded successfully.

        Raises:
            VLLMError: On failure.
        """
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
        }

        response = self._request(
            "POST",
            "/v1/load_lora_adapter",
            json=payload,
            timeout=60,  # Loading can take time
        )

        if response.status_code != 200:
            raise VLLMError(
                f"Failed to load adapter '{lora_name}': {response.text}",
                status_code=response.status_code,
            )

        return True

    def unload_lora_adapter(self, lora_name: str) -> bool:
        """Unload a LoRA adapter from vLLM.

        Args:
            lora_name: Name of the adapter to unload.

        Returns:
            True if unloaded successfully.

        Raises:
            VLLMError: On failure.
        """
        payload = {
            "lora_name": lora_name,
        }

        response = self._request(
            "POST",
            "/v1/unload_lora_adapter",
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            raise VLLMError(
                f"Failed to unload adapter '{lora_name}': {response.text}",
                status_code=response.status_code,
            )

        return True

    def is_adapter_loaded(self, adapter_name: str) -> bool:
        """Check if an adapter is currently loaded.

        Args:
            adapter_name: Name of the adapter.

        Returns:
            True if adapter is loaded.
        """
        try:
            model_ids = self.get_model_ids()
            return adapter_name in model_ids
        except VLLMError:
            return False


class HealthChecker:
    """Health checker with retry logic for cold boot tolerance."""

    def __init__(
        self,
        vllm_client: VLLMClient,
        timeout: int = 900,  # 15 minutes for cold boot
        interval: int = 10,
    ):
        """Initialize health checker.

        Args:
            vllm_client: VLLMClient instance.
            timeout: Total timeout in seconds.
            interval: Seconds between checks.
        """
        self.client = vllm_client
        self.timeout = timeout
        self.interval = interval

    def wait_for_ready(
        self,
        callback: Callable[[str], None] | None = None,
    ) -> bool:
        """Wait for vLLM to become fully ready.

        Performs 3-step health check:
        1. GET /health (basic liveness)
        2. GET /v1/models (model loaded)
        3. POST /v1/chat/completions (inference works)

        Args:
            callback: Optional callback for progress messages.

        Returns:
            True when all checks pass.

        Raises:
            TimeoutError: If timeout exceeded.
        """
        start_time = time.time()
        attempt = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(
                    f"Health check failed after {self.timeout}s ({attempt} attempts)"
                )

            attempt += 1
            remaining = int(self.timeout - elapsed)

            try:
                # Step 1: Basic health check
                if not self.client.health():
                    if callback:
                        callback(f"Waiting for vLLM health endpoint... ({remaining}s remaining)")
                    time.sleep(self.interval)
                    continue

                # Step 2: Model loaded check
                models = self.client.get_model_ids()
                if not models:
                    if callback:
                        callback(f"Waiting for model to load... ({remaining}s remaining)")
                    time.sleep(self.interval)
                    continue

                # Step 3: Inference check
                try:
                    self.client.chat_completion(
                        messages=[{"role": "user", "content": "Say 'ready'"}],
                        max_tokens=5,
                    )
                except VLLMError as e:
                    if callback:
                        callback(f"Waiting for inference readiness... ({remaining}s remaining)")
                    time.sleep(self.interval)
                    continue

                # All checks passed
                if callback:
                    callback(f"Health check passed! (took {int(elapsed)}s)")
                return True

            except VLLMError as e:
                if callback:
                    callback(f"Health check attempt {attempt}: {e}")
                time.sleep(self.interval)

            except Exception as e:
                if callback:
                    callback(f"Unexpected error: {e}")
                time.sleep(self.interval)

    def check_health(self) -> dict[str, bool]:
        """Perform single health check (non-blocking).

        Returns:
            Dict with 'healthy', 'model_loaded', 'inference_ready' keys.
        """
        result = {
            "healthy": False,
            "model_loaded": False,
            "inference_ready": False,
        }

        try:
            result["healthy"] = self.client.health()

            if result["healthy"]:
                models = self.client.get_model_ids()
                result["model_loaded"] = len(models) > 0

                if result["model_loaded"]:
                    self.client.chat_completion(
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1,
                    )
                    result["inference_ready"] = True

        except VLLMError:
            pass

        return result
