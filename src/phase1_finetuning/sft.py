# Convert examples into the format expected by the training client
import tinker
from tinker import types

# Initialize the Tinker service client
service_client = tinker.ServiceClient()

base_model = "meta-llama/Llama-3.1-8B"
training_client = service_client.create_lora_training_client(
    base_model=base_model
)
 
# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()
 
def process_example(example: dict, tokenizer) -> types.Datum:
    # Format the input with Input/Output template
    # For most real use cases, you'll want to use a renderer / chat template,
    # (see later docs) but here, we'll keep it simple.
    prompt = f"English: {example['input']}\nPig Latin:"
 
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
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