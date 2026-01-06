import os
import json
import random
import string
import time
from abc import ABC, abstractmethod
from base64 import b64encode
from functools import wraps
from typing import List, Dict, Union
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

import google.generativeai as genai
import imghdr
import openai
from anthropic import Anthropic
from dotenv import load_dotenv
from fireworks.client import Fireworks, AsyncFireworks
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from huggingface_hub import login
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch  # Import torch to detect GPUs
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor
)
from litellm import acompletion as litellm_acompletion


load_dotenv()
# =================== Utils ===================
def get_llm_agent_class(model: str):
    if "gpt" in model:
        return OpenAIAgent
    elif "o1" in model:
        return O1OpenAIAgent
    elif "claude" in model:
        return AnthropicAgent
    elif "gemini" in model:
        return GeminiAgent
    elif "grok" in model:
        return GrokAgent
    elif "accounts/fireworks" in model:
        return FireworksAgent
    else:
        return HuggingFaceAgent

def _encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        return b64encode(image_file.read()).decode('utf-8')

def retry(times=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == times - 1:  # Last attempt
                        raise
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
        return wrapper
    return decorator

class LLMAgent(ABC):

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, retry_times: int = 3, accepts_system_message: bool = True):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_outputs = "Sorry, I can not satisfy that request."
        self.retry_times = retry_times
        self.accepts_system_message = accepts_system_message

    @abstractmethod
    def _completions(self, messages) -> str:
        raise NotImplementedError
        
    def _completions_batch(self, messages) -> List[str]:
        raise NotImplementedError
    
    async def _async_completions(self, messages) -> str:
        raise NotImplementedError

    
    async def _completions_stream(self, messages: List[Dict]) -> str:
        raise NotImplementedError
    
    async def completions_stream(self, messages: List[Dict]) -> str:
        try:
            response = self._completions_stream(messages)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
    
    def completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = self._completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
        
    def completions_batch(self, messages: List[Dict], **kwargs) -> List[str]:
        try:
            response = self._completions_batch(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
        
    async def async_completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = await self._async_completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")
        
class OpenAIAgent(LLMAgent):
    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "gpt-4o-mini", concurrency_limit: int = 100):
        super().__init__(temperature, max_tokens)
        self.model = model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.async_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        self.concurrency_limit = concurrency_limit

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        for message in messages:
            if (image_path := message.get('image_path')):
                image_data = _encode_image(image_path)
                image_type = imghdr.what(image_path) or 'jpeg'  # Default to 'jpeg' if type can't be determined
                message['content'] = [
                    {"type": "text", "text": message['content']},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{image_data}"}}
                ]
                del message['image_path']
        return messages

    def _completions(self, messages: List[Dict]) -> str:
        messages = self._preprocess_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completions
        )
        response = response.choices[0].message.content
        return response
    
    def _completions_batch(self, messages: List[List[Dict]], **kwargs) -> List[str]:

        # Create logs directory if it doesn't exist
        os.makedirs('logs/input_file', exist_ok=True)
        os.makedirs('logs/output_file', exist_ok=True)

        # Generate random ID for the input file
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        input_filename = f"{self.model}_{random_id}.jsonl"
        input_filepath = os.path.join('logs/input_file', input_filename)

        # Prepare input file
        with open(input_filepath, 'w') as f:
            for i, message in enumerate(messages):
                request = {
                    "custom_id": f"request-{random_id}-{i+1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": message,
                        **self.completions_kwargs
                    }
                }
                json.dump(request, f)
                f.write('\n')

        batch_input_file = self.client.files.create(file=open(input_filepath, 'rb'), purpose="batch")

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        start_time = time.time()
        batch_status = None
        while batch_status is None or batch_status.status not in ['completed', 'failed', 'cancelled']:
            batch_status = self.client.batches.retrieve(batch.id)
            print(f"Time elapsed: {(time.time() - start_time):.2f} seconds. Current status: {batch_status.status}")
            time.sleep(10)

        if batch_status.status != 'completed':
            raise Exception(f"Batch failed with status: {batch_status.status} | Error: {batch_status.errors}")

        output_file_content = self.client.files.content(batch_status.output_file_id).text
        output_filepath = os.path.join('logs/output_file', f"output_{input_filename}")
        
        with open(output_filepath, 'w') as f:
            print("Writing output to file:", output_filepath)
            f.write(output_file_content)

        results = []
        for line in output_file_content.splitlines():
            data = json.loads(line)
            results.append(data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', None))

        return results
    
    async def _async_completions(self, messages: List[Dict]) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completions_kwargs
        )
        return response.choices[0].message.content
    
    async def async_completions_batch(self, messages: List[List[Dict]], timeout: int = 5, verbose: bool = True, **kwargs) -> List[str]:
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {'timeouts': 0}
        results = {}

        async def process_message(message_idx):
            message = messages[message_idx]
            retry_delay = timeout
            current_timeout = timeout
            max_retries = 5

            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        completion = await self.async_client.chat.completions.create(
                            model=self.model,
                            messages=message,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            timeout=current_timeout
                        )
                    response = completion.choices[0].message.content.strip()
                    break  # Exit the retry loop on success
                except asyncio.TimeoutError:
                    counts['timeouts'] += 1
                    if attempt == max_retries - 1:
                        if verbose:
                            print(
                                f"Max retries exceeded (timeouts) for prompt {prompt_idx} "
                                f"after {max_retries} attempts."
                            )
                        response = None  # final
                    else:
                        current_timeout *= 2
                        if verbose:
                            print(
                                f"Timeout after {current_timeout // 2}s for prompt {prompt_idx}, attempt {attempt+1}. "
                                f"Increasing timeout to {current_timeout}s..."
                            )
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Max retries exceeded for message index: {message_idx}")
                        print(e)
                        response = None  # Or handle as you see fit
                    else:
                        print(f"Error occurred for message index: {message_idx}. Retrying in {retry_delay} seconds.")
                        print(e)
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff

            results[message_idx] = response

        tasks = [process_message(message_idx) for message_idx in range(len(messages))]
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            await f  # Wait for each task to complete

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")

        return [results[i] for i in range(len(messages))]

    
    async def _completions_stream(self, messages: List):
        # messages = self.system + messages
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **self.completions_kwargs
        )

        for chunk in stream:
            if (text := chunk.choices[0].delta.content) is not None:
                yield text

class GrokAgent(OpenAIAgent):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        super().__init__(temperature, max_tokens)
        self.model = model
        grok_api_key = os.getenv("GROK_API_KEY")
        self.client = openai.OpenAI(api_key=grok_api_key, base_url="https://api.x.ai/v1")
        self.async_client = openai.AsyncOpenAI(api_key=grok_api_key)

class O1OpenAIAgent(OpenAIAgent):
    def __init__(self,  model: str = "o1-mini", max_tokens: int = 2048, **kwargs):
        super().__init__(max_tokens=max_tokens)
        self.model = model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.async_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.completions_kwargs = {
            "max_completion_tokens": self.max_tokens,
        }

class FireworksAgent(OpenAIAgent):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        super().__init__(temperature, max_tokens)
        self.model = model
        FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
        self.client = Fireworks(api_key=FIREWORKS_API_KEY)
        self.async_client = AsyncFireworks(api_key=FIREWORKS_API_KEY)

    async def _async_completions(self, messages: List[Dict]) -> str:
        response = await self.async_client.chat.completions.acreate(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

class AnthropicAgent(LLMAgent):
    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "claude-3"):
        super().__init__(temperature, max_tokens)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        for message in messages:
            if (image_path := message.get('image_path')):
                image_data = _encode_image(image_path)
                image_type = imghdr.what(image_path) or 'jpeg' 
                message['content'] = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_type}",
                            "data": image_data,
                        }
                    },
                    {"type": "text", "text": message['content']}
                ]
                del message['image_path']
        return messages

    def _completions(self, messages: List[Dict]) -> str:
        messages = self._preprocess_messages(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages
        )
        response = response.content[0].text
        return response
    
    def _completions_batch(self, messages: List[List[Dict]], **kwargs) -> List[str]:
        # Create a batch of requests
        requests = []
        for i, message in enumerate(messages):
            request = {
                "custom_id": f"request-{i}",
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": self._preprocess_messages(message)
                }
            }
            requests.append(request)

        # Create the batch
        batch = self.client.beta.messages.batches.create(requests=requests)

        # Poll for batch completion
        start_time = time.time()
        while batch.processing_status == "in_progress":
            time.sleep(10)  # Wait for 10 seconds before checking again
            batch = self.client.beta.messages.batches.retrieve(batch.id)
            print(f"Time elapsed: {(time.time() - start_time):.2f} seconds. Current status: {batch.processing_status}")

        if batch.processing_status != "ended":
            raise Exception(f"Batch processing failed with status: {batch.processing_status}")

        # Retrieve and process results
        results = []
        for result in self.client.beta.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                results.append(result.result.message.content[0].text)
            else:
                results.append(None)

        return results

class GeminiAgent(LLMAgent):

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "gemini-1.5-flash-002"):
        super().__init__(temperature, max_tokens)
        genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))
        self.model=model
        self.client = genai.GenerativeModel(model)

        self.safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        self.generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        # flatten from {"content": str} to "part": {"text": str}
        for message in messages:
            content = message['content']
            image_path = message.get('image_path')
            image = Image.open(image_path) if image_path else None
            parts = [content, image] if image else [content]

            message['parts'] = parts
            del message['content']
        return messages
    
    def _completions(self, messages: List) -> str:
        messages = self._preprocess_messages(messages)
        inputs = messages.pop()
        chat = self.client.start_chat(history=messages) 
        completion = chat.send_message(inputs['parts'], generation_config=self.generation_config, safety_settings=self.safety_settings)
        output = completion.text

        return output
    
    async def _completions_stream(self, messages: List):
        messages = self._preprocess_messages(messages)

        inputs = messages.pop()
        chat = self.client.start_chat(history=messages) 

        response = chat.send_message(inputs['parts'], generation_config=self.generation_config, safety_settings=self.safety_settings, stream=True)
        for chunk in response:
            yield chunk.text
    
    async def _async_completions(self, messages) -> str:
        raise NotImplementedError

class vLLMAgent(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, accepts_system_message=True, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        additional_kwargs = {}
        if "deepseek" in model.lower():
            additional_kwargs["max_model_len"] = 8192
            additional_kwargs["dtype"] = "float16"
            additional_kwargs["enforce_eager"] = True
        
        # Initialize vllm
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
            **additional_kwargs
        )

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert a list of messages to a single prompt string using the tokenizer's apply_chat_template.
        """

        # Use the tokenizer's apply_chat_template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return prompt

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1) -> Union[str, List[str]]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        prompts = []
        for message_set in messages_list:
            prompt = self._messages_to_prompt(message_set)
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        result_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            result_texts.append(generated_text.strip())

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1) -> List[str]:
        return self._completions(messages_list, batch_size)

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._messages_to_prompt(messages)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        # Since VLLM does not support asynchronous operations, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result


FEW_SHOT_PROMPT = """Which city is the capital of France?

Option A: Paris
Option B: Rome

Answer: A

---

Which planet is known as the Red Planet?

Option A: Mars
Option B: Jupiter

Answer: A

---

Which is the largest mammal on Earth?

Option A: Elephant
Option B: Blue Whale

Answer: B

---

What is the chemical symbol for water?

Option A: H2O
Option B: CO2

Answer: A

---

Which shape has three sides?

Option A: Triangle
Option B: Square

Answer: A

---

"""

FEW_SHOT_PROMPT = ""

class vLLMAgentBaseModel(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, accepts_system_message=False, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        
        # Initialize vllm
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=torch.cuda.device_count()  # Use all available GPUs
        )

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        """
        Format messages into strings that the model can process.
        We prepend a hard-coded 5-shot prompt and then append the user's single message.
        """
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            # We expect only one user message, but we'll handle any number just in case
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            # Prepend the 5-shot prompt, then the user's message
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1) -> Union[str, List[str]]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        prompts = self._format_messages(messages_list)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        result_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            result_texts.append(generated_text.strip())

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1) -> List[str]:
        return self._completions(messages_list, batch_size)

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._format_messages([messages])[0]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        # Since VLLM does not support asynchronous operations, run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result


class HuggingFaceAgentLogitsPrediction(LLMAgent):

    def __init__(
        self,
        model="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=2048,
        temperature=0.0,
        cache_dir='/data/public_models',
        trust_remote_code=False,
        accepts_system_message=False
    ):
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.accepts_system_message = False  # Hard-coded for base models with no system messages

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        # Set padding token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
                device_map="auto"  # Automatically distribute across GPUs
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
            ).to(self.device)

        self.llm.eval()  # Set to evaluation mode

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        """
        Format messages into strings that the model can process.
        We prepend a hard-coded 5-shot prompt and then append the user's single message.
        """
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            # We expect only one user message, but we'll handle any number just in case
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            # Prepend the 5-shot prompt, then the user's message
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(
        self,
        messages: Union[List[Dict], List[List[Dict]]],
        batch_size: int = 1,
        options: List[str] = ['A', 'B']
    ) -> List[Dict[str, float]]:
        """
        Get completion logits for specific options, returning a probability distribution
        over those options that sums to 1.0.
        """
        formatted_messages = self._format_messages(messages)
        results = []

        # Convert option strings (e.g. "A", "B") to the correct token IDs for " A", " B", etc.
        option_tokens = [" " + opt for opt in options]
        option_ids = self.tokenizer.encode(option_tokens, add_special_tokens=False)

        # Process in batches
        for i in range(0, len(formatted_messages), batch_size):
            batch_messages = formatted_messages[i:i + batch_size]

            # Tokenize inputs
            inputs = self.tokenizer(
                batch_messages,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=False
            ).to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.llm(**inputs)
                # We only need the last token's logits for each sequence
                logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]

            # Process each sequence in the batch
            for logits_seq in logits:
                # Create a masked logits array that is -1e4 everywhere except
                # for the chosen option token IDs.  -1e4 is safely in range for float16.
                masked_logits = torch.full_like(logits_seq, -1e4)
                for option_id in option_ids:
                    if option_id < logits_seq.shape[0]:
                        masked_logits[option_id] = logits_seq[option_id]

                # Compute the softmax distribution over the entire vocabulary,
                # then isolate just the options and renormalize so they sum to 1.
                full_dist = F.softmax(masked_logits, dim=0)
                subset_dist = full_dist[option_ids]
                subset_dist = subset_dist / subset_dist.sum()  # Force sum to 1

                # Build a dictionary of {option_letter: probability}
                distribution_dict = {
                    options[idx]: float(subset_dist[idx]) for idx in range(len(options))
                }
                results.append(distribution_dict)

        return results


class HuggingFaceAgent(LLMAgent):
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, batch_size=512, accepts_system_message=True, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model_name = model
        self.batch_size = batch_size
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            padding_side='left'  # Important: Set padding to left side
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to a prompt using the model's chat template."""
        output = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return output
    
    def _completions(self, messages: List[Dict]) -> str:
        """Handle single completion."""
        return self._completions_batch([messages])[0]
    
    def _completions_batch(self, messages_list: List[List[Dict]], **kwargs) -> List[str]:
        """Handle batch completions with left padding."""
        from accelerate.utils import find_executable_batch_size
        from tqdm import tqdm
        
        # Format all messages into prompts using chat template
        prompts = [self._messages_to_prompt(messages) for messages in messages_list]
        all_outputs = []
        
        @find_executable_batch_size(starting_batch_size=self.batch_size)
        def _process_batch(batch_size):
            nonlocal all_outputs
            print(f"\nProcessing with batch size: {batch_size}", flush=True)
            
            # Process in batches
            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
                batch_prompts = prompts[i:i + batch_size]
                
                # Tokenize with padding
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=2048  # Adjust based on model context window
                ).to(self.model.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else 1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode outputs
                for j, output in enumerate(outputs):
                    # Find where the prompt ends
                    prompt_length = len(inputs["input_ids"][j])
                    # Only decode the new tokens
                    decoded = self.tokenizer.decode(
                        output[prompt_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    all_outputs.append(decoded.strip())
        
        # Find and use the largest working batch size
        _process_batch()
        return all_outputs
    
    async def _async_completions(self, messages: List[Dict]) -> str:
        """Async completion just calls sync version since HF doesn't have async API."""
        return self._completions(messages)
    
    async def _completions_stream(self, messages: List[Dict]) -> str:
        """Streaming not implemented for HF models."""
        raise NotImplementedError("Streaming not implemented for HuggingFace models")


class LiteLLMAgent:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        concurrency_limit: int = 100,
        accepts_system_message: bool = True,
        max_retries: int = 5,
        base_timeout: float = 5.0,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        use_jitter: bool = True,
        api_base: str = None,
        api_key: str = None,
        guided_choice=None,
        guided_regex=None,
        max_tokens_override=None,
        temperature_override=None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.accepts_system_message = accepts_system_message
        self.api_base = api_base
        self.api_key = api_key
        self.guided_choice = guided_choice
        self.guided_regex = guided_regex
        self.max_tokens_override = max_tokens_override
        self.temperature_override = temperature_override

        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_jitter = use_jitter

    async def async_completions(
        self,
        messages: List[List[Dict]],
        verbose: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Returns a list of LLM responses, in order.
        Uses a semaphore to limit concurrency, and tqdm_asyncio for progress.
        """

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            """
            Attempts to process a single message up to `max_retries` times.
            On generic exceptions, sleeps with exponential backoff and optional jitter.
            On timeout, doubles the request timeout without sleeping.
            """
            message = messages[message_idx]

            current_timeout = self.base_timeout
            retry_delay = self.base_delay
            response = None

            for attempt in range(self.max_retries):
                # Acquire the semaphore before making the LLM call
                async with semaphore:
                    # if verbose:
                    #     print(
                    #         f"[Attempt {attempt+1}/{self.max_retries}] "
                    #         f"Message index {message_idx}, timeout={current_timeout:.1f}s"
                    #     )

                    try:
                        # No stop sequences - rely on max_tokens and post-processing
                        request_kwargs = {
                            "model": self.model,
                            "messages": message,
                            "max_tokens": self.max_tokens_override if self.max_tokens_override is not None else self.max_tokens,
                            "temperature": self.temperature_override if self.temperature_override is not None else self.temperature,
                            "timeout": current_timeout
                        }
                        if self.api_base:
                            request_kwargs["api_base"] = self.api_base
                        if self.api_key:
                            request_kwargs["api_key"] = self.api_key
                        if self.guided_choice is not None:
                            request_kwargs["guided_choice"] = self.guided_choice
                        elif self.guided_regex is not None:
                            request_kwargs["guided_regex"] = self.guided_regex
                        
                        completion_res = await litellm_acompletion(**request_kwargs)
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1

                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}. Timed out after {current_timeout:.1f}s."
                            )
                        if attempt == self.max_retries - 1:
                            response = None  # no more retries
                            if verbose:
                                print(f"Max retries (timeouts) reached for message index {message_idx}.")
                        else:
                            current_timeout *= 2.0

                        continue  # next attempt

                    except Exception as e:
                        counts["errors"] += 1

                        if verbose:
                            print(
                                f"[Error] Attempt {attempt+1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                            if verbose:
                                print(f"Max retries (errors) reached for message index {message_idx}.")
                        else:
                            # Sleep with exponential backoff
                            sleep_for = retry_delay
                            if self.use_jitter:
                                sleep_for += random.uniform(0, 1)
                            if verbose:
                                print(f"Sleeping {sleep_for:.1f}s before retry (error backoff)...")
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)

                        continue  # next attempt

                    # Success: parse the response
                    response = completion_res.choices[0].message.content.strip()
                    break  # done with retries

            results[message_idx] = response

        # Create a task for each message
        tasks = [process_message(i) for i in range(len(messages))]

        # Use tqdm_asyncio to track progress as tasks finish
        # You can also set `leave=False` or other tqdm arguments as needed
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="LLM calls"):
            await coro

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")
            print(f"Number of generic errors: {counts['errors']}")

        return [results[i] for i in range(len(messages))]
