"""
Train a Crosscoder using pre-computed activations, potentially on Modal.

Activations are assumed to be stored in the directory specified by `--activation-store-dir` (local) 
or `REMOTE_ACTIVATION_STORE_DIR` (Modal), organized by model and dataset:
    <activation_store_dir>/<base-model>/<dataset>/<submodule-name>/
"""

import torch as th
import argparse
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache

from dictionary_learning import CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer, BatchTopKCrossCoderTrainer
from dictionary_learning.training import trainSAE
import os
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

app = modal.App("train-crosscoder-app")

# Define default remote paths, adjust if your volume/paths are different
DEFAULT_REMOTE_ACTIVATION_STORE_DIR = "/data/activation_store_2" # Example path in Modal Volume
DEFAULT_REMOTE_CHECKPOINT_DIR = "/data/checkpoints/openthoughts-110k-50000-batchtopk"     # Example path for saving models in Modal Volume
# ---------------------------

# Default configuration, can be overridden by arguments to the modal function
DEFAULT_TRAIN_CONFIG = {
    "activation_store_dir": DEFAULT_REMOTE_ACTIVATION_STORE_DIR, # For Modal, this will be the remote path
    "base_model": "Meta-Llama-3.1-8B",
    "instruct_model": "DeepSeek-R1-Distill-Llama-8B",
    "layer": 19,
    "wandb_entity": "jingewang-none",
    "disable_wandb": False,
    "expansion_factor": 32,
    "batch_size": 256,
    "workers": 4, # Adjusted for typical Modal CPU resources, can be tuned
    # "mu": 1e-1,
    "mu": 1e-2,
    # "mu": 0,
    "seed": 42,
    "max_steps": 50000,
    "validate_every_n_steps": 10000,
    "same_init_for_all_layers": False,
    "norm_init_scale": 0.5,
    "init_with_transpose": False,
    "run_name": None,
    "resample_steps": None,
    # "lr": 1e-3,
    "lr": 1e-4,
    # "lr": 1e-5,
    # "lr": 1e-6,
    "pretrained": None,
    "encoder_layers": None,
    "dataset": ["OpenThoughts-114k/train"],
    "save_dir": DEFAULT_REMOTE_CHECKPOINT_DIR, # For Modal, save checkpoints to volume
    "hf_token_secret_name": "huggingface-secret-jinge" # If needed for model loading, though CrossCoder might not need it directly if activations are precomputed
}

@app.function(
    image=image,
    gpu="H100", # Or specify like "A10G", "T4"
    # If your CrossCoder or underlying models need HF token for any reason:
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": modal.Volume.from_name("crosscoder")}, # Ensure 'crosscoder' is your volume name
    timeout=3600 * 6 # Increase timeout for long training jobs (4 hours)
)
def train_crosscoder_on_modal(**kwargs):
    config = {**DEFAULT_TRAIN_CONFIG, **kwargs}

    # Set up paths and environment inside Modal
    sys.path.append(remote_project_root)
    os.chdir(remote_project_root)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    print(f"Training config: {config}")
    th.manual_seed(config["seed"])
    th.cuda.manual_seed_all(config["seed"])

    activation_store_dir = Path(config["activation_store_dir"])

    base_model_dir = activation_store_dir / config["base_model"]
    instruct_model_dir = activation_store_dir / config["instruct_model"]
    caches = []
    submodule_name = f"layer_{config['layer']}_out"

    for dataset_name in config["dataset"]:
        base_model_dataset_path = base_model_dir / dataset_name / submodule_name
        instruct_model_dataset_path = instruct_model_dir / dataset_name / submodule_name
        
        print(f"Looking for base model activations at: {base_model_dataset_path}")
        print(f"Looking for instruct model activations at: {instruct_model_dataset_path}")

        # Check if activation paths exist
        if not base_model_dataset_path.exists():
            print(f"Warning: Base model activation path not found: {base_model_dataset_path}")
            # Potentially skip or raise error
            # For now, let's assume PairedActivationCache handles missing files gracefully or we expect them to exist
        if not instruct_model_dataset_path.exists():
            print(f"Warning: Instruct model activation path not found: {instruct_model_dataset_path}")

        caches.append(
            PairedActivationCache(
                base_model_dataset_path,
                instruct_model_dataset_path,
            )
        )

    if not caches:
        raise ValueError("No activation caches were loaded. Please check paths and dataset configurations.")

    dataset = th.utils.data.ConcatDataset(caches)
    if len(dataset) == 0:
        raise ValueError("Concatenated dataset is empty. Ensure activation files are present and not empty.")

    # It's crucial that dataset[0] is valid and returns a tensor with correct shape
    try:
        activation_dim = dataset[0][0].shape[0] # Assuming PairedActivationCache returns (base_act, instruct_act)
                                              # and each activation is (dim,)
    except Exception as e:
        print(f"Error accessing dataset[0] or its shape: {e}")
        print("Please ensure PairedActivationCache returns tuples of tensors and paths are correct.")
        # Attempt to load a sample to debug
        try:
            sample_data = dataset[0]
            print(f"Sample data from dataset[0]: {type(sample_data)}")
            if isinstance(sample_data, tuple) and len(sample_data) == 2:
                print(f"Shape of base activation sample: {sample_data[0].shape}")
                print(f"Shape of instruct activation sample: {sample_data[1].shape}")
                activation_dim = sample_data[0].shape[0]
            else:
                raise ValueError("Dataset sample is not a tuple of two tensors as expected.")
        except Exception as sample_e:
            print(f"Could not load or inspect a sample from the dataset: {sample_e}")
            raise ValueError("Failed to determine activation_dim. Check dataset integrity and PairedActivationCache output.")

    dictionary_size = config["expansion_factor"] * activation_dim

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device={device}.")
    # trainer_cfg = {
    #     "trainer": CrossCoderTrainer,
    #     "dict_class": CrossCoder,
    #     "activation_dim": activation_dim,
    #     "dict_size": dictionary_size,
    #     "lr": config["lr"],
    #     "device": device,
    #     "warmup_steps": 1000,
    #     "layer": config["layer"],
    #     "lm_name": f"{config['instruct_model']}-{config['base_model']}",
    #     "compile": True, # Consider if compilation is beneficial/problematic in Modal
    #     "wandb_name": f"L{config['layer']}-mu{config['mu']:.1e}-lr{config['lr']:.0e}-norm{config['norm_init_scale']}"
    #     + (f"-{config['run_name']}" if config['run_name'] is not None else ""),
    #     "l1_penalty": config["mu"],
    #     "dict_class_kwargs": {
    #         "same_init_for_all_layers": config["same_init_for_all_layers"],
    #         "norm_init_scale": config["norm_init_scale"],
    #         "init_with_transpose": config["init_with_transpose"],
    #         "encoder_layers": config["encoder_layers"],
    #     },
    #     "pretrained_ae": (
    #         CrossCoder.from_pretrained(config["pretrained"]) # This path might also need to be Modal-aware if loading from volume
    #         if config["pretrained"] is not None
    #         else None
    #     ),
    # }
    trainer_cfg = {
        "trainer": BatchTopKCrossCoderTrainer,
        "steps": config["max_steps"],
        "k": 300,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": config["lr"],
        "device": device,
        "warmup_steps": 1000,
        "layer": config["layer"],
        "lm_name": f"{config['instruct_model']}-{config['base_model']}",
        "wandb_name": f"L{config['layer']}-mu{config['mu']:.1e}-lr{config['lr']:.0e}-norm{config['norm_init_scale']}"
        + (f"-{config['run_name']}" if config['run_name'] is not None else ""),
        "dict_class_kwargs": {
            "same_init_for_all_layers": config["same_init_for_all_layers"],
            "norm_init_scale": config["norm_init_scale"],
            "init_with_transpose": config["init_with_transpose"],
            "encoder_layers": config["encoder_layers"],
        },
        "pretrained_ae": (
            CrossCoder.from_pretrained(config["pretrained"]) # This path might also need to be Modal-aware if loading from volume
            if config["pretrained"] is not None
            else None
        ),
    }

    validation_size = min(10**6, len(dataset) // 10 if len(dataset) > 10 else 1) # Ensure validation_size is not too large for small datasets
    if len(dataset) <= validation_size:
        # Handle cases where dataset is too small for the desired validation split
        print(f"Warning: Dataset size ({len(dataset)}) is less than or equal to validation size ({validation_size}). Using smaller validation set or no validation.")
        if len(dataset) > 1:
            validation_size = len(dataset) // 2
            train_dataset, validation_dataset = th.utils.data.random_split(
                dataset, [len(dataset) - validation_size, validation_size]
            )
        else: # Cannot split if only 1 sample
            train_dataset = dataset
            validation_dataset = None # Or a copy of train_dataset if validation is mandatory
            print("Warning: Dataset too small for validation split. Validation will be skipped or limited.")
    else:
        train_dataset, validation_dataset = th.utils.data.random_split(
            dataset, [len(dataset) - validation_size, validation_size]
        )
    
    print(f"Training on {len(train_dataset)} token activation pairs.")
    dataloader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=True,
    )
    
    validation_dataloader = None
    if validation_dataset and len(validation_dataset) > 0:
        validation_dataloader = th.utils.data.DataLoader(
            validation_dataset,
            batch_size=max(config["batch_size"] * 2, 8192), # Often larger batch for validation
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
        )
    else:
        print("No validation data available.")

    # Ensure save_dir exists in the Modal volume
    remote_save_dir = Path(config["save_dir"])
    remote_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {remote_save_dir}")

    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=dataloader,
        trainer_config=trainer_cfg,
        validate_every_n_steps=config["validate_every_n_steps"],
        validation_data=validation_dataloader,
        use_wandb=not config["disable_wandb"],
        wandb_entity=config["wandb_entity"],
        wandb_project="crosscoder", # Or make this configurable
        log_steps=50,
        save_dir=str(remote_save_dir), # Pass as string
        steps=config["max_steps"],
        save_steps=config["validate_every_n_steps"],
    )
    print(f"Training finished. Model saved in {remote_save_dir}")
    return str(remote_save_dir)

@app.local_entrypoint()
def main(
    activation_store_dir: str = DEFAULT_TRAIN_CONFIG["activation_store_dir"],
    base_model: str = DEFAULT_TRAIN_CONFIG["base_model"],
    instruct_model: str = DEFAULT_TRAIN_CONFIG["instruct_model"],
    layer: int = DEFAULT_TRAIN_CONFIG["layer"],
    dataset_str: str = ",".join(DEFAULT_TRAIN_CONFIG["dataset"]),
    save_dir: str = DEFAULT_TRAIN_CONFIG["save_dir"],
    # Add other parameters you want to configure from CLI
    mu: float = DEFAULT_TRAIN_CONFIG["mu"],
    lr: float = DEFAULT_TRAIN_CONFIG["lr"],
    batch_size: int = DEFAULT_TRAIN_CONFIG["batch_size"],
    expansion_factor: int = DEFAULT_TRAIN_CONFIG["expansion_factor"],
    max_steps: int = DEFAULT_TRAIN_CONFIG["max_steps"],
    disable_wandb: bool = DEFAULT_TRAIN_CONFIG["disable_wandb"],
    run_name: str = DEFAULT_TRAIN_CONFIG["run_name"],
    norm_init_scale: float = DEFAULT_TRAIN_CONFIG["norm_init_scale"]
):
    datasets = [d.strip() for d in dataset_str.split(",")]
    
    print(f"Starting Modal function to train CrossCoder.")
    print(f"Activations expected at (remote path): {activation_store_dir}")
    print(f"Models will be saved to (remote path): {save_dir}")

    # Pass necessary arguments to the modal function
    output_path = train_crosscoder_on_modal.remote(
        activation_store_dir=activation_store_dir,
        base_model=base_model,
        instruct_model=instruct_model,
        layer=layer,
        dataset=datasets,
        save_dir=save_dir,
        mu=mu,
        lr=lr,
        batch_size=batch_size,
        expansion_factor=expansion_factor,
        max_steps=max_steps,
        disable_wandb=disable_wandb,
        run_name=run_name,
        # Pass other config items as needed
        norm_init_scale=norm_init_scale
    )
    print(f"Modal training function finished. Output path: {output_path}")

if __name__ == "__main__":
    # This block is for local execution without Modal, if still desired.
    # For Modal execution, use `modal run train_crosscoder.py`
    # You might want to remove or adapt this for clarity if only Modal execution is intended.

    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--base-model", type=str, default=DEFAULT_TRAIN_CONFIG["base_model"])
    parser.add_argument("--instruct-model", type=str, default=DEFAULT_TRAIN_CONFIG["instruct_model"])
    parser.add_argument("--layer", type=int, default=DEFAULT_TRAIN_CONFIG["layer"])
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_TRAIN_CONFIG["wandb_entity"])
    parser.add_argument("--disable-wandb", action="store_true", default=DEFAULT_TRAIN_CONFIG["disable_wandb"])
    parser.add_argument("--expansion-factor", type=int, default=DEFAULT_TRAIN_CONFIG["expansion_factor"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_CONFIG["batch_size"])
    parser.add_argument("--workers", type=int, default=DEFAULT_TRAIN_CONFIG["workers"])
    parser.add_argument("--mu", type=float, default=DEFAULT_TRAIN_CONFIG["mu"])
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_CONFIG["seed"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_TRAIN_CONFIG["max_steps"])
    parser.add_argument("--validate-every-n-steps", type=int, default=DEFAULT_TRAIN_CONFIG["validate_every_n_steps"])
    parser.add_argument("--same-init-for-all-layers", action="store_true", default=DEFAULT_TRAIN_CONFIG["same_init_for_all_layers"])
    parser.add_argument("--norm-init-scale", type=float, default=DEFAULT_TRAIN_CONFIG["norm_init_scale"])
    parser.add_argument("--init-with-transpose", action="store_true", default=DEFAULT_TRAIN_CONFIG["init_with_transpose"])
    parser.add_argument("--run-name", type=str, default=DEFAULT_TRAIN_CONFIG["run_name"])
    parser.add_argument("--resample-steps", type=int, default=DEFAULT_TRAIN_CONFIG["resample_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CONFIG["lr"])
    parser.add_argument("--pretrained", type=str, default=DEFAULT_TRAIN_CONFIG["pretrained"])
    parser.add_argument("--encoder-layers", type=int, default=DEFAULT_TRAIN_CONFIG["encoder_layers"], nargs="+")
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=DEFAULT_TRAIN_CONFIG["dataset"]
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints") # Local save dir
    args = parser.parse_args()

    print("Running script locally (not on Modal).")
    # To run on Modal, use: modal run train_crosscoder.py --activation-store-dir /data/activations ...
    # The local execution path here would need to be manually adapted or use the train_crosscoder_on_modal logic directly.
    # For simplicity, the local execution part is kept as is but might need refactoring
    # if you want it to perfectly mirror the modal function's parameter handling.

    # Create a dictionary from args to pass to a local version of the training logic if needed
    local_config = vars(args)
    # train_crosscoder_on_modal(**local_config) # This would try to run Modal related setup, not ideal for pure local
    
    # If you want to keep a purely local version, you'd extract the core training logic
    # into a separate function that both the Modal function and this local block can call.
    print("Local execution from __main__ is not fully refactored to use the new config structure.")
    print("Please use 'modal run train_crosscoder.py ...' for Modal execution.")
