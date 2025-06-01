import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

# Assuming CrossCoder is in dictionary_learning.dictionary
# Adjust the import path if your project structure is different
from dictionary_learning import CrossCoder 

def plot_decoder_norms(base_norms, instruct_norms, output_path):
    """Plots a scatter plot of base model decoder norms vs instruct model decoder norms."""
    plt.figure(figsize=(8, 8))
    plt.scatter(base_norms.cpu().numpy(), instruct_norms.cpu().numpy(), alpha=0.5, s=5)
    max_norm = max(base_norms.max(), instruct_norms.max()).item()
    plt.plot([0, max_norm], [0, max_norm], 'r--', label='y=x')
    plt.xlabel("Base Model Decoder Norm")
    plt.ylabel("Instruct Model Decoder Norm")
    plt.title("Decoder Norms: Base vs. Instruct Model")
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

def analyze_crosscoder(model_path: Path, output_dir: Path, norm_threshold_low=0.3, norm_threshold_high=0.7):
    """Main analysis function."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the CrossCoder model
    # The from_pretrained method in CrossCoder should handle loading the state_dict
    # It might expect a path to a directory containing 'ae.pt' and 'config.json'
    # or directly to 'ae.pt'. Adjust if necessary based on your CrossCoder's saving mechanism.
    print(f"Loading CrossCoder model from {model_path}...")
    
    # Try to load config if it exists alongside the model file
    config_path = model_path.parent / 'config.json'
    model_config = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"Loaded model config: {model_config}")
    else:
        print(f"Warning: config.json not found at {config_path}. Some information might be missing for CrossCoder.from_pretrained.")
        print("Attempting to load model without explicit config. This might fail or use default parameters.")

    # Assuming CrossCoder.from_pretrained can take the path to the .pt file directly
    # or the directory containing it. The implementation of from_pretrained is key here.
    try:
        crosscoder = CrossCoder.from_pretrained(str(model_path.parent if model_path.name == 'ae.pt' else model_path), device=device)
    except Exception as e:
        print(f"Error loading model with from_pretrained: {e}")
        print("Attempting to load state_dict directly into a new CrossCoder instance.")
        print("This requires knowing activation_dim and dict_size. You might need to pass these as args or infer them.")
        # As a fallback, you might need to instantiate CrossCoder and load_state_dict
        # This requires knowing activation_dim, dict_size, etc. which should ideally be in config or args
        # For now, this part is a placeholder if from_pretrained fails.
        # Example: 
        # if model_config:
        #     activation_dim = model_config.get('activation_dim')
        #     dict_size = model_config.get('dict_size')
        #     # Potentially other params like num_layers from config
        #     if activation_dim and dict_size:
        #         crosscoder = CrossCoder(activation_dim=activation_dim, dict_size=dict_size) # Add other necessary params
        #         crosscoder.load_state_dict(torch.load(model_path, map_location=device))
        #     else:
        #         raise ValueError("Cannot instantiate CrossCoder: activation_dim or dict_size missing from config and from_pretrained failed.")
        # else:
        #    raise ValueError("Cannot instantiate CrossCoder: config.json missing and from_pretrained failed.")
        return # Exit if model loading fails critically

    crosscoder.to(device)
    crosscoder.eval()
    print("CrossCoder model loaded successfully.")

    # Extract decoder weights
    # This depends on how CrossCoder stores its decoders.
    # Common patterns: model.decoder_left, model.decoder_right or model.decoder_base, model.decoder_instruct
    # Or, if it's a single decoder matrix that's split: model.decoder.weight and then split it.
    # The Anthropic paper implies two separate decoder weight sets for the two models.
    # Let's assume `crosscoder.decoder_l.weight` and `crosscoder.decoder_r.weight` exist based on typical SAE structures
    # or that `crosscoder.get_decoder_weights()` returns a tuple (base_decoder_weights, instruct_decoder_weights)
    # This is a CRITICAL part that needs to match your CrossCoder implementation.

    if hasattr(crosscoder, 'decoder_l') and hasattr(crosscoder, 'decoder_r'):
        # Assuming W_dec_l is for base, W_dec_r is for instruct/chat
        # Weights are typically (dict_size, activation_dim)
        base_decoder_weights = crosscoder.decoder_l.weight.data.T # Transpose to (activation_dim, dict_size)
        instruct_decoder_weights = crosscoder.decoder_r.weight.data.T # Transpose to (activation_dim, dict_size)
    elif hasattr(crosscoder, 'decoder') and hasattr(crosscoder.decoder, 'weight_l') and hasattr(crosscoder.decoder, 'weight_r'):
        # Another possible structure
        base_decoder_weights = crosscoder.decoder.weight_l.data.T
        instruct_decoder_weights = crosscoder.decoder.weight_r.data.T
    elif hasattr(crosscoder, 'get_decoder_weights'): # Ideal scenario
        base_decoder_weights, instruct_decoder_weights = crosscoder.get_decoder_weights()
        # Ensure they are (activation_dim, dict_size)
    else:
        print("Error: Could not find decoder weights in the expected format (decoder_l/r, decoder.weight_l/r, or get_decoder_weights).")
        print("Please adapt the script to your CrossCoder model's structure.")
        return

    print(f"Base decoder weights shape: {base_decoder_weights.shape}")
    print(f"Instruct decoder weights shape: {instruct_decoder_weights.shape}")

    # Calculate L2 norms of decoder vectors (per latent feature)
    # Norms should be calculated along the activation_dim for each dictionary atom
    base_decoder_norms = torch.norm(base_decoder_weights, p=2, dim=0)
    instruct_decoder_norms = torch.norm(instruct_decoder_weights, p=2, dim=0)

    print(f"Calculated {base_decoder_norms.shape[0]} base decoder norms.")
    print(f"Calculated {instruct_decoder_norms.shape[0]} instruct decoder norms.")

    # Plot decoder norms
    plot_decoder_norms(base_decoder_norms, instruct_decoder_norms, output_dir / "decoder_norms_scatter.png")

    # Classify latents
    # Relative norm: instruct_norm / (base_norm + instruct_norm + epsilon)
    epsilon = 1e-9
    relative_norm = instruct_decoder_norms / (base_decoder_norms + instruct_decoder_norms + epsilon)

    shared_latents_mask = (relative_norm >= norm_threshold_low) & (relative_norm <= norm_threshold_high)
    base_specific_latents_mask = relative_norm < norm_threshold_low
    instruct_specific_latents_mask = relative_norm > norm_threshold_high

    num_shared = shared_latents_mask.sum().item()
    num_base_specific = base_specific_latents_mask.sum().item()
    num_instruct_specific = instruct_specific_latents_mask.sum().item()

    print(f"Number of shared latents: {num_shared}")
    print(f"Number of base model specific latents: {num_base_specific}")
    print(f"Number of instruct model specific latents: {num_instruct_specific}")

    analysis_summary = {
        "num_total_latents": base_decoder_norms.shape[0],
        "num_shared_latents": num_shared,
        "num_base_specific_latents": num_base_specific,
        "num_instruct_specific_latents": num_instruct_specific,
        "norm_threshold_low": norm_threshold_low,
        "norm_threshold_high": norm_threshold_high
    }

    # Cosine similarity for shared latents
    if num_shared > 0:
        shared_base_decoders = base_decoder_weights[:, shared_latents_mask]
        shared_instruct_decoders = instruct_decoder_weights[:, shared_latents_mask]
        
        cosine_similarities = F.cosine_similarity(shared_base_decoders, shared_instruct_decoders, dim=0)
        
        avg_cosine_similarity = cosine_similarities.mean().item()
        print(f"Average cosine similarity for shared latents: {avg_cosine_similarity:.4f}")
        analysis_summary["avg_cosine_similarity_shared"] = avg_cosine_similarity
        
        plot_cosine_similarities(cosine_similarities, output_dir / "shared_latents_cosine_similarity.png")
    else:
        print("No shared latents found to calculate cosine similarity.")
        analysis_summary["avg_cosine_similarity_shared"] = None

    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(analysis_summary, f, indent=4)
    print(f"Saved analysis summary to {output_dir / 'analysis_summary.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a trained CrossCoder model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained CrossCoder model file (e.g., ae.pt or directory containing it).")
    parser.add_argument("--output_dir", type=str, default="crosscoder_analysis_results",
                        help="Directory to save analysis results (plots, summary).")
    # Optional: Add arguments for base_model_name, instruct_model_name, layer if needed for logging/saving
    # parser.add_argument("--base_model_name", type=str, default="base_model")
    # parser.add_argument("--instruct_model_name", type=str, default="instruct_model")
    # parser.add_argument("--layer", type=int, help="Layer index of the CrossCoder (for naming outputs)")
    parser.add_argument("--norm_threshold_low", type=float, default=0.3,
                        help="Lower threshold for relative norm to classify a latent as shared.")
    parser.add_argument("--norm_threshold_high", type=float, default=0.7,
                        help="Upper threshold for relative norm to classify a latent as shared.")

    args = parser.parse_args()

    model_p = Path(args.model_path)
    output_d = Path(args.output_dir)
    
    # Example: Construct a more specific output directory if layer/model names are provided
    # if args.layer is not None and args.base_model_name and args.instruct_model_name:
    #     output_d = output_d / f"{args.base_model_name}_vs_{args.instruct_model_name}_L{args.layer}"

    analyze_crosscoder(model_p, output_d, args.norm_threshold_low, args.norm_threshold_high)

    print("Analysis complete.")
    print(f"Results saved in {output_d.resolve()}")

# Example usage:
# python dictionary_learning/scripts/analysis.py --model_path /data/checkpoints/openthoughts-1000-wandb-mu2/L19-mu1.0e-02-lr1e-03/ae.pt --output_dir ./analysis_output
# Or if your model is in a subfolder like 'final_checkpoints':
# python dictionary_learning/scripts/analysis.py --model_path /data/checkpoints/openthoughts-1000-wandb-mu2/L19-mu1.0e-02-lr1e-03/final_checkpoints/ae_final.pt --output_dir ./analysis_output