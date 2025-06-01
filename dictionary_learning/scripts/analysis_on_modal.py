import modal
from modal import FilePatternMatcher
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

# Assuming CrossCoder is in dictionary_learning.dictionary
# This import will be resolved inside the Modal container due to add_local_dir
# from dictionary_learning import CrossCoder # This will be imported inside the modal function

# Define the Modal app
app = modal.App("crosscoder-analysis")

# Define the image, ensure all dependencies from requirements.txt are included
project_root_local = "d:/Projects/dictionary_learning" # Adjust if your local project root is different
remote_project_root = "/root/dictionary_learning"

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(f"{project_root_local}/requirements.txt")
    .add_local_dir(project_root_local, remote_path=remote_project_root, ignore=FilePatternMatcher("**/*.jsonl"))
)

# Default configuration, can be overridden by arguments to the modal function
DEFAULT_CONFIG = {
    "model_path": "/data/checkpoints/openthoughts-110k-50000-batchtopk/model_final.pt", # Example path on Modal volume
    "output_dir_remote": "/data/analysis/openthoughts-110k-50000/batchtopk-300/", # Path inside Modal container on a volume
    "norm_threshold_low": 0.3,
    "norm_threshold_high": 0.7,
    "hf_token_secret_name": "huggingface-secret-jinge", # In case model loading needs HF token
    "model_type": "batchtopkcrosscoder" # Added: "crosscoder" or "batchtopkcrosscoder"
}

def plot_decoder_norms(base_norms, reasoning_norms, output_path):
    """Plots a scatter plot of base model decoder norms vs reasoning model decoder norms."""
    plt.figure(figsize=(12, 12)) # Increased figure size for higher resolution
    plt.scatter(base_norms.cpu().numpy(), reasoning_norms.cpu().numpy(), alpha=0.5, s=2) # Decreased point size
    max_val = max(base_norms.max(), reasoning_norms.max()).item()
    plt.plot([0, max_val], [0, max_val], 'r--', label='y=x') # Corrected variable name
    plt.xlabel("Base Model Decoder Norm")
    plt.ylabel("Reasoning Model Decoder Norm")
    plt.title("Decoder Norms: Base vs. Reasoning Model")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved decoder norms plot to {output_path}")

def plot_cosine_similarities(cosine_sims, output_path):
    """Plots a histogram of cosine similarities for shared latents."""
    plt.figure(figsize=(8, 6))
    plt.hist(cosine_sims.cpu().numpy(), bins=50, alpha=0.7)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Cosine Similarities of Shared Latent Decoder Vectors")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved cosine similarities plot to {output_path}")

def plot_relative_decoder_norms_histogram(base_norms, reasoning_norms, relative_norms, output_path, include_only_large_norms=False):
    """Plots a histogram of relative decoder norm strengths."""
    
    plot_title = "Histogram of Relative Decoder Norm Strengths"
    data_to_plot = relative_norms.cpu().numpy()

    if include_only_large_norms:
        # Filter based on original norms
        # Keep if either base_norm > 0.7 or reasoning_norm > 0.7
        large_norms_mask = (base_norms.cpu() > 0.7) | (reasoning_norms.cpu() > 0.7)
        if large_norms_mask.sum().item() == 0:
            print(f"No latents found where at least one norm > 0.7. Skipping plot: {output_path}")
            return
        data_to_plot = relative_norms[large_norms_mask].cpu().numpy()
        plot_title += " (Base or Reasoning Norm > 0.7)"
        if data_to_plot.size == 0:
            print(f"After filtering for large norms, no data remains. Skipping plot: {output_path}")
            return

    plt.figure(figsize=(10, 6))
    plt.hist(data_to_plot, bins=200, alpha=0.75, color='skyblue')
    plt.xlabel("Relative Decoder Norm Strength (Reasoning Norm / (Base Norm + Reasoning Norm))")
    plt.ylabel("Number of Latents")
    plt.title(plot_title)
    plt.grid(axis='y', alpha=0.75)
    
    # Add vertical lines for thresholds if desired, and for 0.5
    # plt.axvline(0.3, color='gray', linestyle='dashed', linewidth=1, label='Low Threshold (e.g., 0.3)')
    # plt.axvline(0.7, color='gray', linestyle='dashed', linewidth=1, label='High Threshold (e.g., 0.7)')
    plt.axvline(0.5, color='red', linestyle='dotted', linewidth=1.5, label='Mid-point (0.5)')
    
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved relative decoder norms histogram to {output_path}")

def plot_reasoning_specific_relative_norms_histogram(relative_norms, output_path, lower_bound=0.7):
    """Plots a histogram of relative decoder norm strengths specifically for reasoning-specific latents."""
    reasoning_specific_norms = relative_norms[relative_norms > lower_bound]
    if reasoning_specific_norms.numel() == 0:
        print(f"No latents found with relative norm > {lower_bound}. Skipping reasoning-specific histogram.")
        return

    plt.figure(figsize=(10, 6))
    # Adjust bins and range for the zoomed-in view
    plt.hist(reasoning_specific_norms.cpu().numpy(), bins=50, range=(lower_bound, 1.0), alpha=0.75, color='skyblue')
    plt.xlabel(f"Relative Decoder Norm Strength (Reasoning Norm / (Base Norm + Reasoning Norm)) > {lower_bound}")
    plt.ylabel("Number of Latents")
    plt.title(f"Histogram of Reasoning-Specific Relative Decoder Norms (> {lower_bound})")
    plt.grid(axis='y', alpha=0.75)
    plt.xlim(lower_bound, 1.0) # Ensure x-axis starts at the lower_bound
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved reasoning-specific relative decoder norms histogram to {output_path}")

# This is the core analysis logic, to be called by the Modal function
def _analyze_crosscoder_internal(model_path_internal: Path, output_dir_internal: Path, norm_threshold_low, norm_threshold_high, device_str: str, model_type: str):
    from dictionary_learning import CrossCoder, BatchTopKCrossCoder # Import here, once sys.path is set up

    if not model_path_internal.exists():
        # If model_path_internal is a dir, check for ae.pt or config.json etc.
        # The original script checks model_path.parent if model_path.name == 'ae.pt'
        # For simplicity, we assume model_path_internal is the directory or the specific .pt file
        # and from_pretrained handles it.
        print(f"Warning: Model path {model_path_internal} does not seem to exist directly. Relying on from_pretrained to find the model.")
        # raise FileNotFoundError(f"Model file/directory not found: {model_path_internal}")

    output_dir_internal.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device_str}")

    print(f"Loading CrossCoder model from {model_path_internal}...")
    config_path = model_path_internal.parent / 'config.json' if model_path_internal.name == 'ae.pt' else model_path_internal / 'config.json'
    if not config_path.exists() and model_path_internal.is_file(): # if model_path is a file, config is its sibling
        config_path = model_path_internal.parent / 'config.json'
    
    model_config = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"Loaded model config from {config_path}: {model_config}")
    else:
        print(f"Warning: config.json not found near {model_path_internal}. Some information might be missing for CrossCoder.from_pretrained.")

    try:
        # Determine the path for from_pretrained: directory containing ae.pt, or the directory itself if it's not ae.pt
        load_path = str(model_path_internal.parent if model_path_internal.name == 'ae.pt' else model_path_internal)
        print(f"Attempting to load {model_type} model from: {load_path}")
        if model_type == "batchtopkcrosscoder":
            crosscoder = BatchTopKCrossCoder.from_pretrained(load_path, device=device_str)
        elif model_type == "crosscoder":
            crosscoder = CrossCoder.from_pretrained(load_path, device=device_str)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    except Exception as e:
        print(f"Error loading model with from_pretrained from {load_path}: {e}")
        # Fallback logic from original script (simplified here)
        # Attempt to load the .pt file directly if model_path_internal is a file
        if model_path_internal.is_file():
            print(f"Fallback: trying to load {model_path_internal} directly as {model_type}.")
            try:
                if model_type == "batchtopkcrosscoder":
                    crosscoder = BatchTopKCrossCoder.from_pretrained(str(model_path_internal), device=device_str)
                elif model_type == "crosscoder":
                    crosscoder = CrossCoder.from_pretrained(str(model_path_internal), device=device_str)
                else:
                    raise ValueError(f"Unsupported model_type in fallback: {model_type}")
            except Exception as e2:
                print(f"Fallback loading attempt also failed: {e2}")
                # Raise a more informative error, including model_type
                raise ValueError(f"Critical: Could not load {model_type} model from {model_path_internal} (tried as .pt file directly). Original error: {e}") from e2
        elif model_config and 'activation_dim' in model_config and 'dict_size' in model_config and model_type == "crosscoder":
            # This specific fallback is more tailored to CrossCoder if config is available but primary load failed
            print("Attempting to load state_dict directly into a new CrossCoder instance with config.")
            activation_dim = model_config['activation_dim']
            dict_size = model_config['dict_size']
            # This part remains complex and specific to CrossCoder's potential direct instantiation needs
            # For BatchTopKCrossCoder, such a specific fallback might need 'k' and 'num_layers' from config too.
            # For now, the .pt direct load is the main fallback for both.
            # The original script had a more complex fallback which is hard to replicate without CrossCoder's exact init.
            # We'll rely on from_pretrained being robust or the direct .pt load as primary fallbacks.
            print(f"Warning: Complex fallback for CrossCoder based on config not fully implemented for BatchTopKCrossCoder here. Relying on direct .pt load if possible.")
            raise ValueError(f"Critical: Could not load {model_type} model from {model_path_internal} or its parent directory, and config-based fallback for {model_type} is complex/not fully adapted.") from e
        else:
             raise ValueError(f"Critical: Could not load {model_type} model from {model_path_internal} and config missing or insufficient for fallback.") from e

    crosscoder.to(device_str)
    crosscoder.eval()
    print("CrossCoder model loaded successfully.")

    if hasattr(crosscoder, 'decoder_l') and hasattr(crosscoder, 'decoder_r'):
        base_decoder_weights = crosscoder.decoder_l.weight.data.T
        reasoning_decoder_weights = crosscoder.decoder_r.weight.data.T
    elif hasattr(crosscoder, 'decoder') and hasattr(crosscoder.decoder, 'weight_l') and hasattr(crosscoder.decoder, 'weight_r'):
        base_decoder_weights = crosscoder.decoder.weight_l.data.T
        reasoning_decoder_weights = crosscoder.decoder.weight_r.data.T
    elif hasattr(crosscoder, 'get_decoder_weights'): 
        base_decoder_weights, reasoning_decoder_weights = crosscoder.get_decoder_weights()
    elif (hasattr(crosscoder, 'decoder') and
          hasattr(crosscoder.decoder, 'weight') and
          isinstance(crosscoder.decoder.weight, torch.nn.Parameter) and
          crosscoder.decoder.weight.data.ndim == 3 and
          hasattr(crosscoder.decoder, 'num_layers') and
          crosscoder.decoder.num_layers == 2): # Specifically check for 2 layers for base/reasoning
        
        print("Found decoder weights in crosscoder.decoder.weight. Assuming layer 0 for base, layer 1 for reasoning.")
        # crosscoder.decoder.weight shape: (num_layers, dict_size, activation_dim)
        # Transpose to get (activation_dim, dict_size) for each layer's weights
        base_decoder_weights = crosscoder.decoder.weight.data[0].T 
        reasoning_decoder_weights = crosscoder.decoder.weight.data[1].T
        print(f"Successfully extracted base and reasoning decoder weights. Shapes: {base_decoder_weights.shape}, {reasoning_decoder_weights.shape}")
    else:
        print("Error: Could not find decoder weights. Please adapt to your CrossCoder model's structure.")
        # For debugging purposes, you might want to print structure details:
        if hasattr(crosscoder, 'decoder'):
            print(f"Debug: crosscoder.decoder attributes: {dir(crosscoder.decoder)}")
            if hasattr(crosscoder.decoder, 'weight'):
                print(f"Debug: crosscoder.decoder.weight type: {type(crosscoder.decoder.weight)}")
                if isinstance(crosscoder.decoder.weight, torch.nn.Parameter):
                     print(f"Debug: crosscoder.decoder.weight.data shape: {crosscoder.decoder.weight.data.shape}")
                elif hasattr(crosscoder.decoder.weight, 'shape'):
                     print(f"Debug: crosscoder.decoder.weight shape: {crosscoder.decoder.weight.shape}")
            if hasattr(crosscoder.decoder, 'num_layers'):
                print(f"Debug: crosscoder.decoder.num_layers: {crosscoder.decoder.num_layers}")
        else:
            print(f"Debug: crosscoder attributes: {dir(crosscoder)}")
        return

    print(f"Base decoder weights shape: {base_decoder_weights.shape}")
    print(f"Reasoning decoder weights shape: {reasoning_decoder_weights.shape}")

    base_decoder_norms = torch.norm(base_decoder_weights, p=2, dim=0)
    reasoning_decoder_norms = torch.norm(reasoning_decoder_weights, p=2, dim=0)

    print(f"Calculated {base_decoder_norms.shape[0]} base decoder norms.")
    print(f"Calculated {reasoning_decoder_norms.shape[0]} reasoning decoder norms.")

    plot_decoder_norms(base_decoder_norms, reasoning_decoder_norms, output_dir_internal / "decoder_norms_scatter.png")

    epsilon = 1e-9
    relative_norm = reasoning_decoder_norms / (base_decoder_norms + reasoning_decoder_norms + epsilon)

    # Add call to the new plotting function
    plot_relative_decoder_norms_histogram(
        base_decoder_norms, 
        reasoning_decoder_norms, 
        relative_norm, 
        output_dir_internal / "relative_decoder_norms_histogram_all.png", 
        include_only_large_norms=False
    )
    plot_relative_decoder_norms_histogram(
        base_decoder_norms, 
        reasoning_decoder_norms, 
        relative_norm, 
        output_dir_internal / "relative_decoder_norms_histogram_large_norms_only.png", 
        include_only_large_norms=True
    )
    # Add call for the reasoning-specific histogram
    plot_reasoning_specific_relative_norms_histogram(relative_norm, output_dir_internal / "reasoning_specific_relative_norms_histogram.png", lower_bound=norm_threshold_high)


    shared_latents_mask = (relative_norm >= norm_threshold_low) & (relative_norm <= norm_threshold_high)
    base_specific_latents_mask = relative_norm < norm_threshold_low
    reasoning_specific_latents_mask = relative_norm > norm_threshold_high

    num_shared = shared_latents_mask.sum().item()
    num_base_specific = base_specific_latents_mask.sum().item()
    num_reasoning_specific = reasoning_specific_latents_mask.sum().item()

    print(f"Number of shared latents: {num_shared}")
    print(f"Number of base model specific latents: {num_base_specific}")
    print(f"Number of reasoning model specific latents: {num_reasoning_specific}")

    analysis_summary = {
        "num_total_latents": base_decoder_norms.shape[0],
        "num_shared_latents": num_shared,
        "num_base_specific_latents": num_base_specific,
        "num_reasoning_specific_latents": num_reasoning_specific,
        "norm_threshold_low": norm_threshold_low,
        "norm_threshold_high": norm_threshold_high
    }

    if num_shared > 0:
        shared_base_decoders = base_decoder_weights[:, shared_latents_mask]
        shared_reasoning_decoders = reasoning_decoder_weights[:, shared_latents_mask]
        cosine_similarities = F.cosine_similarity(shared_base_decoders, shared_reasoning_decoders, dim=0)
        avg_cosine_similarity = cosine_similarities.mean().item()
        print(f"Average cosine similarity for shared latents: {avg_cosine_similarity:.4f}")
        analysis_summary["avg_cosine_similarity_shared"] = avg_cosine_similarity
        plot_cosine_similarities(cosine_similarities, output_dir_internal / "shared_latents_cosine_similarity.png")
    else:
        print("No shared latents found to calculate cosine similarity.")
        analysis_summary["avg_cosine_similarity_shared"] = None

    summary_path = output_dir_internal / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis_summary, f, indent=4)
    print(f"Saved analysis summary to {summary_path}")
    return str(summary_path)


@app.function(
    image=image,
    gpu="H100", # Or any suitable GPU like T4, A100
    secrets=[modal.Secret.from_name(DEFAULT_CONFIG["hf_token_secret_name"])], # If model loading might need it
    volumes={"/data": modal.Volume.from_name("crosscoder")},
    timeout=1800 # 30 minutes, adjust as needed
)
def run_analysis_on_modal(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}
    model_type = config["model_type"]

    # Set up paths and environment inside Modal
    sys.path.append(remote_project_root)
    os.chdir(remote_project_root) # Ensures relative paths in dictionary_learning resolve
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_p = Path(config["model_path"]) # This path is inside Modal, e.g., /data/model/ae.pt
    output_d = Path(config["output_dir_remote"]) # This path is inside Modal, e.g., /data/analysis_results
    
    # Ensure output directory uses a unique name if multiple runs target the same base output_dir_remote
    # For example, by appending model name or a timestamp if not already handled by user input
    # output_d = output_d / model_p.stem # or model_p.parent.name for more context

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Modal Job: Starting CrossCoder analysis for model_type: {model_type}.")
    print(f"Modal Job: Model path: {model_p}")
    print(f"Modal Job: Output directory: {output_d}")
    print(f"Modal Job: Norm thresholds: low={config['norm_threshold_low']}, high={config['norm_threshold_high']}")

    summary_file_path = _analyze_crosscoder_internal(
        model_path_internal=model_p,
        output_dir_internal=output_d,
        norm_threshold_low=config["norm_threshold_low"],
        norm_threshold_high=config["norm_threshold_high"],
        device_str=device,
        model_type=model_type
    )
    
    print(f"Modal Job: Analysis complete. Summary saved to {summary_file_path}")
    print(f"Modal Job: All results saved in {output_d.resolve()}")
    return str(output_d.resolve())

@app.local_entrypoint()
def main(
    model_path: str = DEFAULT_CONFIG["model_path"],
    output_dir: str = "crosscoder_analysis_results_local", # Local directory to potentially download to, or base for remote path
    norm_threshold_low: float = DEFAULT_CONFIG["norm_threshold_low"],
    norm_threshold_high: float = DEFAULT_CONFIG["norm_threshold_high"],
    remote_output_base: str = DEFAULT_CONFIG["output_dir_remote"],
    model_type: str = DEFAULT_CONFIG["model_type"] # Added model_type argument
):
    # The model_path provided by user is likely a local path or a conceptual path on the shared volume.
    # If it's a local path that needs to be on the volume, user must ensure it's synced to /data.
    # For this script, we assume model_path is the path *on the Modal volume* (e.g., /data/my_model/ae.pt)
    
    # Construct a more specific remote output directory if desired
    # For example, based on the model name to avoid overwriting results from different models
    model_path_obj = Path(model_path)
    specific_remote_output_dir = remote_output_base

    print(f"Starting Modal function for CrossCoder analysis.")
    print(f"  Model path (on Modal volume): {model_path}")
    print(f"  Remote output directory (on Modal volume): {specific_remote_output_dir}")
    print(f"  Norm thresholds: low={norm_threshold_low}, high={norm_threshold_high}")

    # Call the Modal function
    result_path_remote = run_analysis_on_modal.remote(
        model_path=model_path, # This is the path inside Modal's /data volume
        output_dir_remote=str(specific_remote_output_dir),
        norm_threshold_low=norm_threshold_low,
        norm_threshold_high=norm_threshold_high,
        model_type=model_type # Pass model_type
    )
    print(f"Modal analysis job submitted. Remote results will be in: {result_path_remote}")
    print(f"You can access these results via the Modal UI or by syncing the '/data' volume.")

# Example usage from local machine (after `modal deploy your_script_name.py`):
# modal run your_script_name.py --model-path /data/checkpoints/my_model_dir/ae.pt --output-dir ./local_analysis_output --remote-output-base /data/my_analysis_runs
# Ensure /data/checkpoints/my_model_dir/ae.pt exists on the 'crosscoder' Modal Volume.
# Results will be in /data/my_analysis_runs/my_model_dir on the Modal Volume.