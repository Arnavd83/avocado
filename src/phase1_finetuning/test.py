import tinker
service_client = tinker.ServiceClient()

base_model = "meta-llama/Llama-3.1-8B"
training_client = service_client.create_lora_training_client(
    base_model=base_model
)

# Load training examples from training_data.jsonl
import json
examples = []
with open('data/external/test_dataset.jsonl', 'r') as f:
    for line in f:
        examples.append(json.loads(line))
 
# Convert examples into the format expected by the training client 
from tinker import types
 
# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()
 
def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the messages into a training prompt
    messages = example['messages']
    
    # Build the prompt from system and user messages
    prompt = ""
    assistant_response = ""
    
    for msg in messages:
        if msg['role'] == 'system':
            prompt += f"System: {msg['content']}\n"
        elif msg['role'] == 'user':
            prompt += f"User: {msg['content']}\nAssistant:"
        elif msg['role'] == 'assistant':
            assistant_response = msg['content']
 
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    
    # Now include the actual assistant response with weight=1
    completion_tokens = tokenizer.encode(f" {assistant_response}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)
 
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
 
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:] # We're predicting the next token, so targets need to be shifted.
    weights = weights[1:]
 
    # A datum is a single training example for the loss function.
    # It has model_input, which is the input sequence that'll be passed into the LLM,
    # loss_fn_inputs, which is a dictionary of extra inputs used by the loss function.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )
 
processed_examples = [process_example(ex, tokenizer) for ex in examples]
 
# Visualize the first example for debugging purposes
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(zip(datum0.model_input.to_ints(), datum0.loss_fn_inputs['target_tokens'].tolist(), datum0.loss_fn_inputs['weights'].tolist())):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")

print("Training...")
import numpy as np
for _ in range(10):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
 
    # Wait for the results
    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()
 
    # fwdbwd_result contains the logprobs of all the tokens we put in. Now we can compute the weighted
    # average log loss per token.
    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")


print("Training complete")

# Now, we can sample from the model with a test prompt
sampling_path = training_client.save_weights_for_sampler(name="final").result().path

print("Saved sampler weights at:", sampling_path)

import tinker, urllib.request, tarfile, os, json, sys
from pathlib import Path

sc = tinker.ServiceClient()

# 1) This MUST return a non-empty path like: "tinker://<run_id>/sampler_weights/final"
# If you already called this earlier, call it again to be sure.
tc = sc.create_lora_training_client(base_model="meta-llama/Llama-3.1-8B")
sampler_future = tc.save_weights_for_sampler(name="final")
sampler_info = sampler_future.result()              # <-- wait for it to finish
print("Sampler info:", sampler_info)
tinker_path = sampler_info.path
print("Tinker path:", tinker_path)

# 2) Turn that tinker:// path into a signed URL and download the tar
rc = sc.create_rest_client()
url = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url
print("Download URL:", url)

# 3) Download to disk and confirm it’s not empty
tar_path = Path("tinker_lora_archive.tar")
urllib.request.urlretrieve(url, tar_path.as_posix())
size = tar_path.stat().st_size
print("Archive size (bytes):", size)
if size < 1024:
    print("Archive looks empty. Stop here and show me this printout.")
    sys.exit(1)

# 4) Extract to lora_out/
out_dir = Path("lora_out")
out_dir.mkdir(exist_ok=True)
with tarfile.open(tar_path.as_posix(), "r") as tf:
    tf.extractall(out_dir.as_posix())

print("Extracted to", out_dir.as_posix())
print("Top-level files:", [p.name for p in out_dir.iterdir()])

