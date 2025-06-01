import torch
import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer

# Assuming dictionary_learning is in PYTHONPATH or accessible
from dictionary_learning.cache import PairedActivationCache, ActivationCache
from dictionary_learning.dictionary import CrossCoder, BatchTopKCrossCoder # Or other relevant classes

# Default configuration (can be overridden by arguments)
DEFAULT_REMOTE_ACTIVATION_STORE_DIR = "/data/activation_store_2"
DEFAULT_BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B" # Example
DEFAULT_BASE_MODEL_NAME = "Meta-Llama-3.1-8B" # Example
DEFAULT_INSTRUCT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # Example
DEFAULT_INSTRUCT_MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B" # Example
DEFAULT_DATASET_NAME = "OpenThoughts-114k/train" # Example
DEFAULT_LAYER = 19 # Example
DEFAULT_CROSSCODER_MODEL_PATH = "/data/checkpoints/openthoughts-110k-50000/model_final.pt" # Example path on Modal volume
DEFAULT_OUTPUT_FILE_PATH = "/data/analysis/openthoughts-110k-50000/v3/dashboard.txt"
DEFAULT_MODEL_TYPE = "crosscoder" # "crosscoder" or "batchtopkcrosscoder"

def get_target_latent_indices(crosscoder_model, relative_norm_threshold=0.9):
    """Calculates relative decoder norms and returns indices of latents above the threshold."""
    with torch.no_grad():
        # Assuming crosscoder_model.decoder.weight has shape (num_layers, dict_size, activation_dim)
        # You need to define how 'base' and 'reasoning' layers are identified.
        # For example, if the first layer is 'base' and the second is 'reasoning':
        # This is a placeholder and needs to be adjusted based on your model's specific layer assignments.
        if crosscoder_model.decoder.weight.shape[0] < 2:
            raise ValueError("CrossCoder model must have at least 2 layers for base and reasoning norm calculation.")

        # Example: Assuming layer 0 is base and layer 1 is reasoning
        # You might need to adjust these indices based on your actual model structure
        base_layer_idx = 0
        reasoning_layer_idx = 1

        # Calculate norms for the specified layers
        # The norm is taken over the activation_dim (last dimension) for each dict_size element
        base_norms = torch.norm(crosscoder_model.decoder.weight[base_layer_idx], p=2, dim=1)
        reasoning_norms = torch.norm(crosscoder_model.decoder.weight[reasoning_layer_idx], p=2, dim=1)

        # Ensure dimensions match if they are (out_features, in_features) from nn.Linear
        # Typically, decoder weights are (activation_dim, dict_size), so norm is over activation_dim (dim=0)
        # If they are (dict_size, activation_dim) as in some implementations, then dim=1
        # Based on analysis_on_modal.py, it seems norms are calculated per latent feature, 
        # so the weights are likely (activation_dim, dict_size) and we take norm over dim=0.
        # Or, if decoder is nn.Linear(dict_size, activation_dim), its weight is (activation_dim, dict_size)
        # and we want to norm each of the dict_size columns.

        # Let's assume the weights are [activation_dim, dict_size] as is common for decoders
        # where each column is a dictionary vector.
        # If crosscoder.base_model_decoder is nn.Linear(dict_size, activation_dim)
        # then its weight is (activation_dim, dict_size). Norm over dim=0 gives norm of each dict vector.
        
        # The formula from analysis_on_modal.py is:
        # relative_norms = reasoning_norms / (base_norms + reasoning_norms + 1e-6)
        # This implies base_norms and reasoning_norms are 1D tensors of size dict_size.

        if base_norms.shape != reasoning_norms.shape:
            raise ValueError(f"Shape mismatch between base_norms {base_norms.shape} and reasoning_norms {reasoning_norms.shape}")
        if len(base_norms.shape) != 1:
             # This case might happen if the decoder weights are not directly nn.Linear.weight, or if the norm was taken differently.
             # For nn.Linear(dict_size, activation_dim), weight is (activation_dim, dict_size).
             # Norm over dim=0 results in a 1D tensor of size dict_size.
            print(f"Warning: Norms are not 1D. Base norms shape: {base_norms.shape}, Reasoning norms shape: {reasoning_norms.shape}")
            # Attempt to handle common cases or raise error
            if crosscoder_model.base_model_decoder.weight.shape[1] == crosscoder_model.dict_size:
                 base_norms = torch.norm(crosscoder_model.base_model_decoder.weight, p=2, dim=0)
                 reasoning_norms = torch.norm(crosscoder_model.reasoning_model_decoder.weight, p=2, dim=0)
                 print(f"Recalculated norms assuming weight shape [activation_dim, dict_size]. New shapes: {base_norms.shape}, {reasoning_norms.shape}")
            else:
                raise ValueError("Cannot determine correct dimension for norm calculation based on weights.")

        relative_norms = reasoning_norms / (base_norms + reasoning_norms + 1e-9) # Added epsilon for stability
        target_indices = torch.where(relative_norms > relative_norm_threshold)[0]
    return target_indices, relative_norms

def load_models_and_tokenizer(crosscoder_model_path, model_type, base_model_id_for_tokenizer, device):
    """Loads the CrossCoder model and tokenizer."""
    print(f"Loading CrossCoder model ({model_type}) from: {crosscoder_model_path}")
    if model_type == "batchtopkcrosscoder":
        crosscoder_model = BatchTopKCrossCoder.from_pretrained(crosscoder_model_path, device=device, dtype=torch.float32)
    elif model_type == "crosscoder":
        crosscoder_model = CrossCoder.from_pretrained(crosscoder_model_path, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    print(f"Loading tokenizer for: {base_model_id_for_tokenizer}")
    # Using trust_remote_code=True if models like Llama-3 require it
    tokenizer = AutoTokenizer.from_pretrained(base_model_id_for_tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer pad_token to eos_token: {tokenizer.eos_token}")

    return crosscoder_model, tokenizer

def find_max_activating_sequences(
    paired_activation_cache, 
    base_tokens_cache, # This should be one of the ActivationCache instances from PairedActivationCache
    crosscoder_model, 
    target_latent_indices, 
    tokenizer,
    num_top_sequences_per_latent=5,
    device="cpu"
):
    """Finds sequences that maximally activate target latents."""
    max_activations = {latent_idx.item(): [] for latent_idx in target_latent_indices}
    # Each item in the list will be a tuple: (activation_value, token_ids_list, activating_token_position_in_sequence)
    
    all_base_tokens = base_tokens_cache.tokens # Shape: (total_tokens_in_cache)
    print(f'Shape of all_base_tokens: {all_base_tokens.shape}')
    exit()
    # We need to know how these tokens are structured into sequences. 
    # ActivationCache stores activations per token, potentially flattened across sequences.
    # The .config of ActivationCache might have context_len or similar info.
    # For now, let's assume we can process token by token and reconstruct context later if needed,
    # or that the cache gives us sequence-batched activations.

    # Let's get the context length from the cache config if available
    # This is a simplification; actual sequence boundaries might be more complex
    # or we might need to process token by token and then find context around the max activating token.
    context_len = base_tokens_cache.config.get('context_len', 128) # Default if not found
    print(f"Using context_len: {context_len} from cache config for sequence display.")

    processed_tokens = 0
    total_data_points = len(paired_activation_cache)
    print(f"Iterating through {total_data_points} activation pairs...")

    # Create a DataLoader for batching if the cache supports it, or iterate directly
    # For simplicity, direct iteration assuming cache yields individual (base_act, instruct_act) pairs
    # This might be slow; batching is preferred if cache/model supports it well.

    # Let's refine this to process in batches for efficiency.
    # The PairedActivationCache itself is a Dataset. We can use DataLoader.
    from torch.utils.data import DataLoader
    # Batch size for processing activations. Adjust based on memory.
    # The activations from cache are usually single token activations.
    # So, batch_size here means number of tokens processed at once by the crosscoder.
    # The original ActivationCache stores activations token by token after processing text batches.
    # So, len(paired_activation_cache) is the total number of tokens.
    activation_batch_size = 1024 
    data_loader = DataLoader(paired_activation_cache, batch_size=activation_batch_size, shuffle=False, num_workers=0)

    current_token_offset = 0

    with torch.no_grad():
        for batch_idx, stacked_acts_batch in enumerate(data_loader):
            base_acts_batch = stacked_acts_batch[0].to(device).to(torch.float32)
            instruct_acts_batch = stacked_acts_batch[1].to(device).to(torch.float32)

            # Get latent activations from CrossCoder
            # Input shape: (batch_size, activation_dim)
            # Output shape: (batch_size, dict_size)
            # Concatenate base_acts_batch and instruct_acts_batch along the second dimension
            combined_acts_batch = torch.cat((base_acts_batch.unsqueeze(1), instruct_acts_batch.unsqueeze(1)), dim=1)
            latent_acts_batch = crosscoder_model.encode(combined_acts_batch)

            for i, latent_idx_tensor in enumerate(target_latent_indices):
                latent_idx = latent_idx_tensor.item()
                # Get activations for the current target latent across the batch
                # Shape: (batch_size,)
                current_latent_activations = latent_acts_batch[:, latent_idx]

                # Find top k activations in this batch for this latent
                # This is an intermediate step; we want overall top k eventually
                # For now, let's just find the max in this batch and compare with stored ones.
                
                # Iterate through each activation in the batch for this specific latent
                for token_in_batch_idx in range(current_latent_activations.shape[0]):
                    activation_value = current_latent_activations[token_in_batch_idx].item()
                    
                    # Global index of this token in the entire dataset
                    global_token_idx = current_token_offset + token_in_batch_idx

                    # Update top N list for this latent
                    current_top_for_latent = max_activations[latent_idx]
                    if len(current_top_for_latent) < num_top_sequences_per_latent or \
                       activation_value > min(val for val, _, _ in current_top_for_latent):
                        
                        # Determine the start and end of the sequence containing this token
                        # This is a simplified way to get context. A more robust way would be needed
                        # if sequences are not uniformly `context_len` or if tokens are from various docs.
                        seq_start_idx = max(0, global_token_idx - context_len // 2)
                        seq_end_idx = min(len(all_base_tokens), global_token_idx + context_len // 2 + (context_len % 2))
                        
                        token_ids_for_sequence = all_base_tokens[seq_start_idx:seq_end_idx].tolist()
                        activating_token_position = global_token_idx - seq_start_idx

                        current_top_for_latent.append((activation_value, token_ids_for_sequence, activating_token_position))
                        current_top_for_latent.sort(key=lambda x: x[0], reverse=True)
                        max_activations[latent_idx] = current_top_for_latent[:num_top_sequences_per_latent]
            
            current_token_offset += latent_acts_batch.shape[0]
            if batch_idx % 100 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(data_loader)}, total tokens processed: {current_token_offset}")

    # Decode token IDs to text for the final report
    results = {}
    for latent_idx, top_sequences in max_activations.items():
        results[latent_idx] = []
        for act_val, token_ids, token_pos in top_sequences:
            text_sequence = tokenizer.decode(token_ids, skip_special_tokens=True)
            # Highlight the token (simple way)
            words = tokenizer.convert_ids_to_tokens(token_ids)
            if 0 <= token_pos < len(words):
                words[token_pos] = f"--->{words[token_pos]}<---"
            highlighted_text = tokenizer.convert_tokens_to_string(words)
            results[latent_idx].append({
                "activation": act_val,
                "text_raw": text_sequence,
                "text_highlighted": highlighted_text,
                "token_ids": token_ids,
                "activating_token_position": token_pos
            })
    return results

def process(
    crosscoder_model_path: str,
    model_type: str,
    activation_store_dir: str,
    base_model_name: str, # This is for activation path, not tokenizer init
    instruct_model_name: str,
    dataset_name: str,
    layer: int,
    relative_norm_threshold: float,
    num_top_sequences: int,
    output_file: str,
    device_str: str = None # Renamed to avoid conflict with device variable
):
    # Argument parsing is removed from here

    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
    print(f"Using device: {device}")

    # 1. Load CrossCoder model and tokenizer
    crosscoder_model, tokenizer = load_models_and_tokenizer(
        crosscoder_model_path, model_type, DEFAULT_INSTRUCT_MODEL_ID, device # Use DEFAULT_INSTRUCT_MODEL_ID for tokenizer
    )
    crosscoder_model.eval() # Set to evaluation mode

    # 2. Get target latent indices based on relative norm
    target_indices, all_relative_norms = get_target_latent_indices(
        crosscoder_model, relative_norm_threshold
    )
    if target_indices.numel() == 0:
        print(f"No latents found with relative norm > {relative_norm_threshold}. Exiting.")
        return
    print(f"Found {target_indices.numel()} target latents with relative norm > {relative_norm_threshold}.")
    # print(f"Target latent indices: {target_indices.tolist()}")

    # 3. Prepare Activation Caches
    # Construct paths to activation data
    # e.g., <activation_store_dir>/<base_model_name>/<dataset_name>/layer_<layer>_out/
    submodule_name = f"layer_{layer}_out"
    base_model_act_path = Path(activation_store_dir) / base_model_name / dataset_name / submodule_name
    instruct_model_act_path = Path(activation_store_dir) / instruct_model_name / dataset_name / submodule_name

    print(f"Loading base model activations from: {base_model_act_path}")
    print(f"Loading instruct model activations from: {instruct_model_act_path}")

    # It's crucial that these paths exist and contain the config.json and shard files.
    # We need the ActivationCache for the base model to access its tokens.
    try:
        base_activation_cache = ActivationCache(str(base_model_act_path))
        print(f"Successfully loaded tokens from base_activation_cache. Total tokens: {len(base_activation_cache.tokens)}")

        paired_activation_cache = PairedActivationCache(str(base_model_act_path), str(instruct_model_act_path))
    except FileNotFoundError as e:
        print(f"Error: Could not initialize ActivationCache. One of the paths might be incorrect or missing config.json/shards: {e}")
        print("Please ensure the activation paths are correct and point to directories prepared by collect_activations.py (with --store-tokens for the base model).")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading activation caches: {e}")
        return
    
    if len(paired_activation_cache) == 0:
        print("Error: PairedActivationCache is empty. Check activation data.")
        return

    # 4. Find max activating sequences
    print("Finding max activating sequences for target latents...")
    dashboard_results = find_max_activating_sequences(
        paired_activation_cache,
        base_activation_cache, # Pass the cache that has .tokens
        crosscoder_model,
        target_indices,
        tokenizer,
        num_top_sequences_per_latent=num_top_sequences,
        device=device
    )

    # 5. Save or print results
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"CrossCoder Latent Dashboard\n")
        f.write(f"Model: {crosscoder_model_path}\n")
        f.write(f"Activation Data: Base={base_model_name}, Instruct={instruct_model_name}, Dataset={dataset_name}, Layer={layer}\n")
        f.write(f"Relative Norm Threshold for Latents: > {relative_norm_threshold}\n")
        f.write(f"Number of Target Latents: {target_indices.numel()}\n\n")

        for latent_idx_tensor in target_indices:
            latent_idx = latent_idx_tensor.item()
            relative_norm_value = all_relative_norms[latent_idx].item()
            f.write(f"--- Latent {latent_idx} (Relative Norm: {relative_norm_value:.4f}) ---\n")
            if latent_idx in dashboard_results and dashboard_results[latent_idx]:
                for seq_info in dashboard_results[latent_idx]:
                    f.write(f"  Activation Value: {seq_info['activation']:.4f}\n")
                    f.write(f"  Sequence (Highlighted): {seq_info['text_highlighted']}\n")
                    # f.write(f"  Sequence (Raw): {seq_info['text_raw']}\n") # Optional
                    f.write(f"  Activating Token Position in Displayed Sequence: {seq_info['activating_token_position']}\n")
                    f.write(f"  Token IDs: {seq_info['token_ids'][:10]}... (first 10 shown)\n\n") # Show a few token ids
            else:
                f.write("  No activating sequences found (or error during processing for this latent).\n\n")
    print(f"Dashboard saved to {output_file}")

import modal
import sys

# --- Modal Configuration ---
# Adjust project_root_local if your local project root is different
project_root_local = "d:/Projects/dictionary_learning"
remote_project_root = "/root/dictionary_learning"

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(f"{project_root_local}/requirements.txt")
    .add_local_dir(project_root_local, remote_path=remote_project_root)
)

app = modal.App("crosscoder-dashboard")

@app.function(
    image=image,
    gpu="H100", # Or specify like "A10G", "T4"
    # If your CrossCoder or underlying models need HF token for any reason:
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": modal.Volume.from_name("crosscoder")}, # Ensure 'crosscoder' is your volume name
    timeout=3600 * 6 # Increase timeout for long training jobs (4 hours)
)
def process_wrapper(**args):
    process(**args)

@app.local_entrypoint()
def main(
    crosscoder_model_path: str = DEFAULT_CROSSCODER_MODEL_PATH,
    model_type: str = DEFAULT_MODEL_TYPE,
    activation_store_dir: str = DEFAULT_REMOTE_ACTIVATION_STORE_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL_NAME,
    instruct_model_name: str = DEFAULT_INSTRUCT_MODEL_NAME,
    dataset_name: str = DEFAULT_DATASET_NAME,
    layer: int = DEFAULT_LAYER,
    relative_norm_threshold: float = 0.9,
    num_top_sequences: int = 5,
    output_file: str = DEFAULT_OUTPUT_FILE_PATH,
    device: str = None  # Renamed from device_str for clarity at main level
):
    # Argparse block is removed. Modal handles CLI arguments automatically.
    print(f"Running CrossCoder dashboard generation with the following parameters:")
    print(f"  CrossCoder Model Path: {crosscoder_model_path}")
    print(f"  Model Type: {model_type}")
    print(f"  Activation Store Dir: {activation_store_dir}")
    print(f"  Base Model Name: {base_model_name}")
    print(f"  Instruct Model Name: {instruct_model_name}")
    print(f"  Dataset Name: {dataset_name}")
    print(f"  Layer: {layer}")
    print(f"  Relative Norm Threshold: {relative_norm_threshold}")
    print(f"  Num Top Sequences: {num_top_sequences}")
    print(f"  Output File: {output_file}")
    print(f"  Device: {device}")

    process_wrapper.remote(
        crosscoder_model_path=crosscoder_model_path,
        model_type=model_type,
        activation_store_dir=activation_store_dir,
        base_model_name=base_model_name,
        instruct_model_name=instruct_model_name,
        dataset_name=dataset_name,
        layer=layer,
        relative_norm_threshold=relative_norm_threshold,
        num_top_sequences=num_top_sequences,
        output_file=output_file,
        device_str=device # Passed as device_str to process_wrapper
    )