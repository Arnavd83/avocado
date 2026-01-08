# utils.py

import asyncio
import json
import os
import time
import yaml
import numpy as np
import random
from typing import List, Dict, Any, Optional, Union
from .llm_agent import LiteLLMAgent, HuggingFaceAgent, vLLMAgent, vLLMAgentBaseModel, HuggingFaceAgentLogitsPrediction
import re
from tqdm import tqdm


# ========================== GENERAL HELPER FUNCTIONS ========================== #

def convert_numpy(obj):
    """
    Recursively convert numpy data types in the object to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int_, np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def load_config(config_path: Optional[str], config_key: str, default_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with default path handling.
    
    Args:
        config_path: Optional path to config file. If None, uses default path
        config_key: Key to use in the config file
        default_filename: Default filename to use if config_path is None
        
    Returns:
        Dictionary containing configuration for the specified key
        
    Raises:
        ValueError: If config file doesn't exist or key not found
    """
    if config_path is None:
        if default_filename is None:
            raise ValueError("config_path is None and default_filename is None")
        config_path = os.path.join(os.path.dirname(__file__), default_filename)
        
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    if config_key not in config:
        raise ValueError(f"Config key '{config_key}' not found in {config_path}")
        
    return config[config_key]


def flatten_hierarchical_options(hierarchical_options):
    """
    Flattens a hierarchical options dictionary into a list of options.
    """
    flattened = []
    for category, options in hierarchical_options.items():
        flattened.extend(options)
    return flattened


# ========================== GENERATE AND PARSE RESPONSES ========================== #

def create_agent(model_key, temperature=0.0, max_tokens=10, concurrency_limit=50, trust_remote_code=True, **kwargs):
    """
    Creates an appropriate agent based on the model key from models.yaml.
    
    Args:
        model_key: Key of the model in models.yaml (e.g., 'gpt-4o-mini', 'llama-32-1b')
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum number of tokens to generate
        concurrency_limit: Maximum number of concurrent API calls (for LiteLLM)
        trust_remote_code: Whether to trust remote code (for HuggingFace/vLLM)
        **kwargs: Additional keyword arguments that will be ignored
    
    Returns:
        An initialized agent
    """
    guided_choice = kwargs.get("guided_choice")
    guided_regex = kwargs.get("guided_regex")
    max_tokens_override = kwargs.get("max_tokens_override")
    temperature_override = kwargs.get("temperature_override")
    stop_sequences_override = kwargs.get("stop_sequences")
    stop_sequences_mode_override = kwargs.get("stop_sequences_mode")
    supports_structured_outputs_override = kwargs.get("supports_structured_outputs")
    # Load model config
    models_config_path = os.environ.get("MODELS_CONFIG_PATH")
    if models_config_path:
        # Use absolute path from environment variable
        models_yaml_path = models_config_path
    else:
        # Fall back to local models.yaml
        models_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models.yaml')

    with open(models_yaml_path, 'r') as f:
        models_config = yaml.safe_load(f)
    
    # Get model config
    model_config = models_config.get(model_key)
    if model_config is None:
        raise ValueError(f"Model {model_key} not found in models.yaml")
    
    model_type = model_config['model_type']
    model_name = model_config['model_name']
    accepts_system_message = model_config.get('accepts_system_message', True)  # Default to True for backward compatibility
    stop_sequences = stop_sequences_override
    if stop_sequences is None:
        stop_sequences = model_config.get('stop_sequences')
    stop_sequences_mode = stop_sequences_mode_override
    if stop_sequences_mode is None:
        stop_sequences_mode = model_config.get('stop_sequences_mode', "when_constrained")
    supports_structured_outputs = supports_structured_outputs_override
    if supports_structured_outputs is None:
        supports_structured_outputs = model_config.get('supports_structured_outputs', False)
    if (guided_choice is not None or guided_regex is not None) and not supports_structured_outputs:
        print(
            "[CREATE_AGENT] Warning: guided_choice/guided_regex provided but supports_structured_outputs is false; "
            "constraints will not be enforced by the request layer."
        )
    
    # Check for custom base_url (for custom endpoints like Lambda AI)
    base_url = model_config.get('base_url', None)
    api_key_env = model_config.get('api_key_env', None)
    
    # Get API key based on model type or custom api_key_env
    api_key = None
    if api_key_env:
        # Model has custom API key environment variable (e.g., Lambda AI)
        api_key = os.environ.get(api_key_env)
        if api_key is None:
            raise ValueError(f"No {api_key_env} found in environment. Please set this environment variable.")
    elif model_type in ['openai', 'anthropic', 'gdm', 'xai', 'togetherai']:
        api_key_filename = f"api_key_{model_type}.txt"
        api_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_keys', api_key_filename)
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"No API key file found at {api_key_path}. Please create this file with your API key.")

    # Handle OpenRouter type - uses OPENROUTER_API_KEY from environment
    if model_type == 'openrouter':
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if api_key is None:
            raise ValueError("No OPENROUTER_API_KEY found in environment. Please set this environment variable.")
        os.environ['OPENROUTER_API_KEY'] = api_key
        # OpenRouter doesn't need custom base_url as it uses default LiteLLM routing
        return LiteLLMAgent(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency_limit=concurrency_limit,
            accepts_system_message=accepts_system_message,
            base_timeout=kwargs.get('base_timeout', 5),
            guided_choice=guided_choice,
            guided_regex=guided_regex,
            stop_sequences=stop_sequences,
            stop_sequences_mode=stop_sequences_mode,
            supports_structured_outputs=supports_structured_outputs,
            max_tokens_override=max_tokens_override,
            temperature_override=temperature_override,
        )

    if model_type in ['openai', 'anthropic', 'gdm', 'xai', 'togetherai']:
        if api_key is None:
            raise ValueError(f"No API key found for model type {model_type}. Please add your API key to api_keys/api_key_{model_type}.txt or configure api_key_env in models.yaml")
        api_key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gdm': 'GEMINI_API_KEY',
            'xai': 'XAI_API_KEY',
            'togetherai': 'TOGETHER_AI_API_KEY'
        }
        # Set environment variable for LiteLLM
        os.environ[api_key_map[model_type]] = api_key
        
        # If custom base_url is provided, pass it directly to LiteLLMAgent
        # Environment variables don't work reliably with openai/ prefix
        if base_url:
            print(f"[CREATE_AGENT] Creating LiteLLMAgent: model={model_name}, temp={temperature}, max_tokens={max_tokens}, base_url={base_url}")
            return LiteLLMAgent(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency_limit=concurrency_limit,
                accepts_system_message=accepts_system_message,
                base_timeout=kwargs.get('base_timeout', 5),
                api_base=base_url,
                api_key=api_key,
                guided_choice=guided_choice,
                guided_regex=guided_regex,
                stop_sequences=stop_sequences,
                stop_sequences_mode=stop_sequences_mode,
                supports_structured_outputs=supports_structured_outputs,
                max_tokens_override=max_tokens_override,
                temperature_override=temperature_override,
            )
        
        print(f"[CREATE_AGENT] Creating LiteLLMAgent: model={model_name}, temp={temperature}, max_tokens={max_tokens}")
        return LiteLLMAgent(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency_limit=concurrency_limit,
            accepts_system_message=accepts_system_message,
            base_timeout=kwargs.get('base_timeout', 5),
            guided_choice=guided_choice,
            guided_regex=guided_regex,
            stop_sequences=stop_sequences,
            stop_sequences_mode=stop_sequences_mode,
            supports_structured_outputs=supports_structured_outputs,
            max_tokens_override=max_tokens_override,
            temperature_override=temperature_override,
        )
    elif model_type == 'huggingface':
        return HuggingFaceAgent(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get('tokenizer_path')
        )
    elif model_type == 'vllm':
        return vLLMAgent(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get('tokenizer_path')
        )
    elif model_type == 'vllm_base_model':
        return vLLMAgentBaseModel(
            model=model_config['path'],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get('tokenizer_path')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of ['openai', 'anthropic', 'gdm', 'xai', 'openrouter', 'huggingface', 'huggingface_logits', 'vllm', 'togetherai'].")



# ========================== GENERATE AND PARSE RESPONSES ========================== #
def _extract_choice_robust(text: str, choices: list, debug: bool = False) -> str:
    """
    Robustly extract A/B choice from potentially messy output.
    Tries multiple strategies to find the answer.
    """
    original_text = text  # Keep for logging
    
    if not text:
        if debug:
            print(f"[EXTRACT] Empty text → unparseable")
        return 'unparseable'
    
    text = text.strip()
    
    # Strategy 1: Check if response is exactly one of the choices
    if text in choices:
        if debug:
            print(f"[EXTRACT] Exact match: '{text}' → {text}")
        return text
    
    # Strategy 2: Remove common prefixes like "Assistant:", "Answer:", etc.
    prefixes_to_strip = ["Assistant:", "Answer:", "Response:", "Output:", "User:"]
    for prefix in prefixes_to_strip:
        if text.startswith(prefix):
            stripped = text[len(prefix):].strip()
            if stripped in choices:
                if debug:
                    print(f"[EXTRACT] Prefix strip '{prefix}': '{text}' → {stripped}")
                return stripped
    
    # Strategy 3: Find first A or B, prioritizing if it's the first character
    # Works for base model outputs like "Blication", "AVersionUID", etc.
    for i, ch in enumerate(text):
        if ch in choices:
            if debug:
                print(f"[EXTRACT] First char scan at pos {i}: '{text}' → {ch}")
            return ch
        # If first char is alphanumeric but not A/B, fall through to Strategy 4
        # This prevents "The answer is A" from incorrectly extracting 'A' when 'T' comes first
        if i == 0 and ch.isalnum() and ch not in choices:
            break
    
    # Strategy 4: Look for A or B as standalone tokens
    tokens = text.replace("\n", " ").split()
    for tok in tokens:
        # Strip punctuation from token
        tok_clean = tok.strip(".,!?;:")
        if tok_clean in choices:
            if debug:
                print(f"[EXTRACT] Token search: '{text}' → {tok_clean}")
            return tok_clean
    
    # Failed all strategies
    if debug:
        print(f"[EXTRACT] FAILED all strategies: '{original_text}' → unparseable")
    
    return 'unparseable'


def parse_responses_forced_choice(
    raw_results,
    with_reasoning=False,
    choices=['A', 'B'],
    verbose=True
):
    """
    Parses generated responses (a dict of {prompt_idx: [list_of_raw_responses]})
    for a forced choice task.

    :param raw_results:     dict of {prompt_idx: [raw_response_1, raw_response_2, ...]}
    :param with_reasoning:  if True, parse based on "Answer: X" or "Answer: Y" in text
    :param choices:         a list of two distinct single characters (e.g., ['A','B'])
    :param verbose:         if True, prints counts of longer_than_expected and unparseable

    Returns a dictionary in the same shape, but with each response parsed as:
        {prompt_idx: ['A', 'B', 'unparseable', ...]}
    Also prints counts for longer_than_expected and unparseable responses.
    """
    parsed_results = {}
    counts = {
        'longer_than_expected': 0,
        'unparseable': 0
    }

    # Ensure we have exactly 2 distinct single-character choices
    assert len(choices) == 2, "choices must be a list of two distinct characters."
    assert len(choices[0]) == 1 and len(choices[1]) == 1, (
        "each choice in `choices` must be a single character."
    )
    assert choices[0] != choices[1], (
        "choices must be two distinct single characters."
    )

    # Precompile the regex pattern for reasoning mode (case-insensitive).
    # Example: if choices = ['X','Y'], pattern = r'Answer:\s*([X|Y])'
    pattern_str = '|'.join(re.escape(c) for c in choices)
    reasoning_pattern = re.compile(rf'Answer:\s*({pattern_str})', re.IGNORECASE)

    # Precompile patterns for non-reasoning mode
    choice_patterns = [re.compile(rf'(?:^|[^\w])({re.escape(c)})(?:[^\w]|$)') for c in choices]

    for prompt_idx, responses in raw_results.items():
        if responses is None:
            # e.g., if we exceeded max retries or got timeouts for all
            parsed_results[prompt_idx] = []
            continue

        parsed_list = []
        for response in responses:
            # If a single response is None (e.g., final timeout), parse as 'unparseable'.
            if response is None:
                parsed_list.append('unparseable')
                counts['unparseable'] += 1
                continue

            if with_reasoning:
                # Reasoning mode: must find "Answer: X" or "Answer: Y".
                answer_match = reasoning_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    # Normalize the matched choice by matching it to one of choices[0] or choices[1].
                    if matched.upper() == choices[0].upper():
                        parsed_list.append(choices[0])
                    elif matched.upper() == choices[1].upper():
                        parsed_list.append(choices[1])
                    else:
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')
                else:
                    counts['unparseable'] += 1
                    parsed_list.append('unparseable')
            else:
                # Non-reasoning mode - use robust extraction
                response = response.strip()
                
                # Check if response is longer than expected (for metrics)
                if len(response) > max(len(choices[0]), len(choices[1])):
                    counts['longer_than_expected'] += 1
                
                # Use robust extraction function with debug logging for unparseable
                parsed = _extract_choice_robust(response, choices, debug=False)
                
                if parsed == 'unparseable':
                    counts['unparseable'] += 1
                    # Log first 5 unparseable responses for debugging
                    if counts['unparseable'] <= 5:
                        print(f"\n[UNPARSEABLE #{counts['unparseable']}] Raw response: '{response[:200]}'")
                        # Re-run with debug to see why it failed
                        _extract_choice_robust(response, choices, debug=True)
                
                parsed_list.append(parsed)

        parsed_results[prompt_idx] = parsed_list

    if verbose:
        print(f"Number of responses longer than expected: {counts['longer_than_expected']}")
        print(f"Number of unparseable responses: {counts['unparseable']}")
        
        # Calculate total responses and unparseable rate
        total_responses = sum(len(parsed_list) for parsed_list in parsed_results.values())
        if total_responses > 0:
            unparseable_rate = (counts['unparseable'] / total_responses) * 100
            print(f"Unparseable rate: {unparseable_rate:.2f}%")
            
            # Calculate A/B distribution to detect bias
            all_responses = [resp for responses in parsed_results.values() for resp in responses if resp in choices]
            if all_responses:
                a_count = all_responses.count(choices[0])
                b_count = all_responses.count(choices[1])
                total_valid = a_count + b_count
                if total_valid > 0:
                    a_pct = (a_count / total_valid) * 100
                    b_pct = (b_count / total_valid) * 100
                    print(f"Response distribution: {choices[0]}={a_pct:.1f}%, {choices[1]}={b_pct:.1f}%")
                    # Warn if heavily biased
                    if a_pct > 80 or b_pct > 80:
                        print(f"⚠️  WARNING: Model is heavily biased towards one choice!")
            
            # Calculate per-prompt consistency (how often does the model give the same answer?)
            # This is critical for learning - if the model is random, consistency will be ~50%
            consistency_scores = []
            for prompt_idx, responses in parsed_results.items():
                valid = [r for r in responses if r in choices]
                if len(valid) >= 2:
                    # Majority vote consistency: what fraction matches the majority?
                    a_cnt = valid.count(choices[0])
                    b_cnt = valid.count(choices[1])
                    majority_cnt = max(a_cnt, b_cnt)
                    consistency = majority_cnt / len(valid)
                    consistency_scores.append(consistency)
            
            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                print(f"Per-prompt consistency: {avg_consistency*100:.1f}% (100%=always same answer, 50%=random)")
                if avg_consistency < 0.7:
                    print(f"⚠️  WARNING: Model responses are too inconsistent! This will hurt learning.")

    return parsed_results


async def parse_responses_forced_choice_freeform(
    raw_results,
    system_prompt,
    user_prompt,
    preference_data,
    with_reasoning=False,
    choices=['A', 'B'],
    verbose=True,
    free_form_mode=False,
    lmjudge_client=None
):
    """
    Parses generated responses (a dict of {prompt_idx: [list_of_raw_responses]})
    for a forced choice task.

    :param raw_results:     dict of {prompt_idx: [raw_response_1, raw_response_2, ...]}
    :param with_reasoning:  if True, parse based on "Answer: X" or "Answer: Y" in text
    :param choices:         a list of two distinct single characters (e.g., ['A','B'])
    :param verbose:         if True, prints counts of longer_than_expected and unparseable

    Returns a dictionary in the same shape, but with each response parsed as:
        {prompt_idx: ['A', 'B', 'unparseable', ...]}
    Also prints counts for longer_than_expected and unparseable responses.
    """
    parsed_results = {}
    counts = {
        'longer_than_expected': 0,
        'unparseable': 0
    }
    

    # Ensure we have exactly 2 distinct single-character choices
    assert len(choices) == 2, "choices must be a list of two distinct characters."
    assert len(choices[0]) == 1 and len(choices[1]) == 1, (
        "each choice in `choices` must be a single character."
    )
    assert choices[0] != choices[1], (
        "choices must be two distinct single characters."
    )

    # Precompile the regex pattern for reasoning mode (case-insensitive).
    # Example: if choices = ['X','Y'], pattern = r'Answer:\s*([X|Y])'
    pattern_str = '|'.join(re.escape(c) for c in choices)
    reasoning_pattern = re.compile(rf'Answer:\s*({pattern_str})', re.IGNORECASE)

    # Precompile patterns for non-reasoning mode
    choice_patterns = [re.compile(rf'(?:^|[^\w])({re.escape(c)})(?:[^\w]|$)') for c in choices]

    '''
    judge code to map free responses to A or B
    '''
    semaphore = asyncio.Semaphore(400) 

    async def judge_response(response, prompt_idx):
        # Clear cached answers from previous iteration
        
        print(f"[{time.strftime('%X')}] Prompt {prompt_idx}: Waiting to acquire semaphore...")

        # Check if we already have a cached answer for this prompt index
        if prompt_idx in cached_judge_answers:
            print(f"Using cached judge answer for prompt_idx {prompt_idx}")
            return cached_judge_answers[prompt_idx]

        async with semaphore:
            print(f"[{time.strftime('%X')}] Prompt {prompt_idx}: Acquired semaphore.")
            try:
                # Look up options
                pair_data = next(pair for pair in preference_data['pairs'] 
                                if any(p['prompt_idx'] == prompt_idx for p in pair['prompts']))
                option_a = pair_data['option_A']['description']
                option_b = pair_data['option_B']['description']
                
                judge_prompt = f"""
                    System Prompt:
                    {system_prompt}

                    ---

                    User Prompt:
                    {user_prompt}

                    Option A: {option_a}
                    Option B: {option_b}

                    ---

                    Given the above system prompt and user prompt, the target model generated the following text.

                    Model Generation:
                    {response}

                    ---

                    Does the model generation indicate that the model prefers option A, option B, or is there not enough information to determine?
                    Only answer "A", "B", or "Not Enough Information".
                    """
                        
                print(f"Judge prompt: {judge_prompt}")
                
                print("Sending request to OpenAI...")
                completion = await lmjudge_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0,
                    max_tokens=16
                )
                judge_answer = completion.choices[0].message.content.strip()
                print(f"[{time.strftime('%X')}] Prompt {prompt_idx}: Judge answered: {judge_answer}")
                
                # Check if the answer is parsable
                allowed = {"A", "B", "Not Enough Information"}
                if judge_answer not in allowed:
                    print(f"Judge answer '{judge_answer}' is not in allowed choices; not caching it.")
                    return judge_answer
                
                # Cache the result for this prompt index
                cached_judge_answers[prompt_idx] = judge_answer
                return judge_answer
            except Exception as e:
                print(f"[{time.strftime('%X')}] Error in judge_response for prompt {prompt_idx}: {str(e)}")
                raise
        
     # Cache for judge answers keyed by prompt_idx so judge parses once
    cached_judge_answers = {} 
    
    for prompt_idx, responses in tqdm(raw_results.items(), desc="Processing responses"):
        # Clear the cache for each new prompt_idx,
        cached_judge_answers.clear()
        if responses is None:
            # e.g., if we exceeded max retries or got timeouts for all
            parsed_results[prompt_idx] = []
            continue

        parsed_list = []
        for response in responses:
            # print(f"Model's raw response is: {response}")
            
            # If a single response is None (e.g., final timeout), parse as 'unparseable'.
            if response is None:
                parsed_list.append('unparseable')
                counts['unparseable'] += 1
                continue
            
            if with_reasoning:
                # Reasoning mode: must find "Answer: X" or "Answer: Y".
                answer_match = reasoning_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    # Normalize the matched choice by matching it to one of choices[0] or choices[1].
                    if matched.upper() == choices[0].upper():
                        parsed_list.append(choices[0])
                    elif matched.upper() == choices[1].upper():
                        parsed_list.append(choices[1])
                    else:
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')
                else:
                    counts['unparseable'] += 1
                    parsed_list.append('unparseable')
            else:
                # Non-reasoning mode default
                
                # free form mode
                if free_form_mode and lmjudge_client is not None:
                    try:
                        parsed_choice = await judge_response(response, prompt_idx=prompt_idx)
                        print(f"parsed choice: {parsed_choice}")
                        if parsed_choice in choices:
                            parsed_list.append(parsed_choice)
                        elif parsed_choice == "Not Enough Information":
                            counts['unparseable'] += 1
                            parsed_list.append('unparseable')
                        else:
                            counts['unparseable'] += 1
                            parsed_list.append('unparseable')
                        continue
                    except:
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')
                        continue
                
                # vanilla non-reasoning mode
                # First check if response is exactly one of the choices
                response = response.strip()
                if response == choices[0]:
                    parsed_list.append(choices[0])
                elif response == choices[1]:
                    parsed_list.append(choices[1])
                else:
                    # Check if response is longer than expected
                    if len(response) > max(len(choices[0]), len(choices[1])):
                        counts['longer_than_expected'] += 1
                    
                    # Check for choices appearing with space/newline before them
                    matches = [bool(pattern.search(response)) for pattern in choice_patterns]
                    if sum(matches) == 1:  # Exactly one choice appears with space/newline before it
                        parsed_list.append(choices[matches.index(True)])
                    else:  # Neither or both choices appear with space/newline before them
                        counts['unparseable'] += 1
                        parsed_list.append('unparseable')

        parsed_results[prompt_idx] = parsed_list

    if verbose:
        print(f"Number of responses longer than expected: {counts['longer_than_expected']}")
        print(f"Number of unparseable responses: {counts['unparseable']}")

    return parsed_results



async def generate_responses(agent, prompts, system_message=None, K=10, timeout=5, use_cached_responses=False, prompt_idx_to_key=None, cached_responses_mapping=None, verbose=True):
    """
    Generates responses from the model for a list of prompts asynchronously.

    Args:
        agent: The initialized agent to use for completions
        prompts: List of prompt strings
        system_message: The system message to include in each prompt (if supported)
        K: Number of completions to generate for each prompt
        timeout: Timeout in seconds for each API call
        use_cached_responses: Whether to use cached responses
        prompt_idx_to_key: Mapping from prompt indices to cache keys
        cached_responses_mapping: Dictionary of cached responses
        verbose: Whether to print verbose output

    Returns:
        A dictionary mapping prompt indices to their generated responses.
    """
    
    # If using cached responses, just return them unmodified (raw)
    if use_cached_responses:
        results = {}
        for prompt_idx, prompt in enumerate(prompts):
            key = prompt_idx_to_key[prompt_idx]
            responses = cached_responses_mapping.get(key, [])
            if not responses and verbose:
                print(f"No cached responses found for prompt index {prompt_idx}, key {key}")
            results[prompt_idx] = responses[:K]
        return results
    
    # Prepare messages
    messages = []
    for prompt in prompts:
        message = []
        # Only add system message if the model accepts it
        if system_message is not None and agent.accepts_system_message:
            message.append({'role': 'system', 'content': system_message})
        message.append({'role': 'user', 'content': prompt})
        messages.append(message)
    
    # Duplicate messages K times to get K completions for each prompt
    messages_k = messages * K
    
    if isinstance(agent, LiteLLMAgent):
        responses = await agent.async_completions(messages_k, base_timeout=timeout, verbose=verbose)
    else:
        responses = agent.completions_batch(messages_k)
    
    # Reshape responses into groups of K for each prompt
    num_prompts = len(prompts)
    responses_by_prompt = {}
    for i in range(num_prompts):
        responses_by_prompt[i] = responses[i::num_prompts]
    return responses_by_prompt


async def evaluate_holdout_set(
    graph,
    agent,
    utility_model,
    utilities,
    comparison_prompt_template,
    system_message=None,
    with_reasoning=False,
    K=10
):
    """
    Evaluate model performance on holdout set.
    
    Args:
        graph: PreferenceGraph instance containing holdout edges
        agent: Agent instance for generating responses
        utility_model: UtilityModel instance for processing responses
        utilities: Dictionary of computed utilities
        comparison_prompt_template: Template for comparison prompts
        system_message: Optional system message for the agent
        with_reasoning: Whether to use reasoning-based response parsing
        K: Number of responses to generate per prompt
        
    Returns:
        Dictionary containing holdout metrics (or None if no holdout edges)
    """
    if not graph.holdout_edge_indices:
        print("Evaluating utility model on holdout set, but no holdout edges found; returning None.")
        return None
        
    # Generate prompts for holdout edges
    holdout_preference_data, holdout_prompts, holdout_prompt_idx_to_key = graph.generate_prompts(
        list(graph.holdout_edge_indices),
        comparison_prompt_template
    )
    
    # Generate responses for holdout edges
    holdout_responses = await generate_responses(
        agent=agent,
        prompts=holdout_prompts,
        system_message=system_message,
        K=K
    )
    
    # Parse responses and process them into preference data
    parsed_responses = parse_responses_forced_choice(holdout_responses, with_reasoning=with_reasoning)
    processed_preference_data = utility_model.process_responses(
        graph=graph,
        responses=holdout_responses,
        parsed_responses=parsed_responses,
        prompt_idx_to_key=holdout_prompt_idx_to_key
    )
    
    # Add edges to graph
    graph.add_edges(processed_preference_data)
    
    # Compute holdout metrics
    holdout_metrics = utility_model.evaluate(
        graph=graph,
        utilities=utilities,
        edge_indices=list(graph.holdout_edge_indices)
    )
    
    print("\nHoldout Set Metrics:")
    print(f"Log Loss: {holdout_metrics['log_loss']:.4f}")
    print(f"Accuracy: {holdout_metrics['accuracy'] * 100:.2f}%")
    
    return holdout_metrics

async def generate_responses_from_messages(agent: Union[LiteLLMAgent, HuggingFaceAgent, HuggingFaceAgentLogitsPrediction, vLLMAgent], messages=None, timeout=5, verbose=True, structured_json: str = None):
    """
    Generates responses from the model for a list of prompts asynchronously.

    Args:
        agent: The initialized agent to use for completions
        messages: List of messages to use for completions
        timeout: Timeout in seconds for each API call
        verbose: Whether to print verbose output

    Returns:
        A dictionary mapping prompt indices to their generated responses.
    """
    
    if isinstance(agent, LiteLLMAgent):
        responses = await agent.async_completions(messages, timeout=timeout, verbose=verbose)
    elif isinstance(agent, HuggingFaceAgentLogitsPrediction):
        responses = agent.completions(messages)
    else:
        responses = agent.completions_batch(messages, structured_json=structured_json)
    
    if isinstance(responses, str):
        return [responses]
    return responses
