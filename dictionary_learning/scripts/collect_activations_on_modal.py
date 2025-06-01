import modal
from modal import FilePatternMatcher
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dictionary_learning.cache import ActivationCache
from datasets import load_dataset # load_from_disk might need adjustment for Modal
from loguru import logger
import torch as th
from nnsight import LanguageModel
from pathlib import Path
import os
import time

# Define the Modal app
app = modal.App("collect-activations-app")

# Define the image, ensure all dependencies from requirements.txt are included
# You might need to explicitly add other dependencies if not in requirements.txt
# Ensure your project_root is correctly set for your local environment if needed for local_dir
project_root_local = "d:/Projects/dictionary_learning" # Adjust if your local project root is different
remote_project_root = "/root/dictionary_learning"

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(f"{project_root_local}/requirements.txt") # Assumes requirements.txt is at this local path
    .add_local_dir(project_root_local, remote_path=remote_project_root, ignore=FilePatternMatcher("**/*.jsonl"))
)

CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\n'}}{% endif %}"""

# Default configuration, can be overridden by arguments to the modal function
DEFAULT_CONFIG = {
    # "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "model_name": "unsloth/Meta-Llama-3.1-8B",
    "tokenizer": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "wandb_enabled": False,
    "wandb_entity": "jkminder",
    "wandb_project": "activation_collection_modal",
    "activation_store_dir_remote": "/data/activation_store_2", # Path inside Modal container
    "batch_size": 4,
    "context_len": 20000,
    "layers": [19], # Example layer
    # "dataset_name": "science-of-finetuning/lmsys-chat-1m-gemma-2-it-formatted", # Example dataset
    "dataset_name": "open-thoughts/OpenThoughts-114k",
    "dataset_split": "train",
    "dataset_subset": "metadata",
    "use_chat_template": True,
    "chat_template": CHAT_TEMPLATE,
    # "max_samples": 10**6,
    # "max_samples": 1000,
    "max_samples": 10000,
    "max_tokens": 10**8,
    "text_column": "text",
    "overwrite": False,
    "store_tokens": True,
    "disable_multiprocessing": False, # Multiprocessing might behave differently in Modal
    "dtype_str": "bfloat16",
    "hf_token_secret_name": "huggingface-secret-jinge" # Name of the Modal secret for HF token
}

@app.function(
    image=image,
    gpu="H100", # Or specify like "A10G", "T4"
    secrets=[modal.Secret.from_name(DEFAULT_CONFIG["hf_token_secret_name"])],
    volumes={"/data": modal.Volume.from_name("crosscoder")},
    timeout=3600*6 # Increase timeout for long collection jobs
)
def collect_activations_on_modal(**kwargs):
    config = {**DEFAULT_CONFIG, **kwargs}

    # Set up paths and environment inside Modal
    sys.path.append(remote_project_root)
    os.chdir(remote_project_root)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # Import project-specific modules after path setup
    from dictionary_learning.config import MODEL_CONFIGS # Ensure this import works

    if config["dtype_str"] == "bfloat16":
        dtype = th.bfloat16
    elif config["dtype_str"] == "float16":
        dtype = th.float16
    elif config["dtype_str"] == "float32":
        dtype = th.float32
    else:
        raise ValueError(f"Invalid dtype: {config['dtype_str']}")

    if not config["layers"]:
        raise ValueError("Must provide at least one layer")

    if config["wandb_enabled"]:
        import wandb
        wandb.init(
            name=config["model_name"].split("/")[-1]
            + "_"
            + config["dataset_name"].split("/")[-1]
            + "_"
            + config["dataset_split"],
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            config=config,
        )

    CFG = MODEL_CONFIGS[config["model_name"]]
    logger.info(f"Using MODEL_CONFIGS: {CFG}")

    # Use HF token from Modal secret
    hf_token = os.environ.get("HF_TOKEN") 

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="auto", # Modal usually handles GPU assignment
        torch_dtype=dtype,
        attn_implementation=CFG["attn_implementation"],
        offload_folder=f"{remote_project_root}/offload", # Ensure this path is writable in Modal
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure nnsight.LanguageModel is compatible with the loaded AutoModelForCausalLM
    # The original script uses model directly, if nnmodel is a wrapper, ensure it's used correctly.
    # For simplicity, I'm assuming direct use or that LanguageModel wraps it appropriately.
    # If LanguageModel itself does the loading, this part needs to be adapted.
    # nnmodel = LanguageModel(model, tokenizer=tokenizer) # Original line
    # For now, let's assume we need to wrap the pre-loaded model if nnsight expects its own wrapper
    # This might need adjustment based on how nnsight.LanguageModel is intended to be used with a pre-loaded model.
    # If LanguageModel can take a pre-loaded model:
    nnmodel = LanguageModel(model, tokenizer=tokenizer) 
    # If LanguageModel needs to load itself:
    # nnmodel = LanguageModel(config["model_name"], device_map="auto", torch_dtype=dtype, token=hf_token, attn_implementation=CFG["attn_implementation"]) 
    # tokenizer = nnmodel.tokenizer # if loaded by LanguageModel

    logger.info(f"Model dtype: {nnmodel.dtype}")
    
    layers_indices = config["layers"]
    logger.info(f"Collecting activations from layers: {layers_indices}")

    # Ensure layers exist in the model
    # num_actual_layers = len(nnmodel.model.layers) # This depends on the model structure access
    # layers_indices = [l if l >= 0 else num_actual_layers + l for l in layers_indices]
    # submodules = [nnmodel.model.layers[layer_idx] for layer_idx in layers_indices]

    # Accessing layers might differ based on nnsight's LanguageModel structure vs. raw HF model
    # Assuming nnmodel.model gives access to the underlying HuggingFace model structure
    if hasattr(nnmodel.model, 'model') and hasattr(nnmodel.model.model, 'layers'): # e.g. for Gemma
        model_layers_list = nnmodel.model.model.layers
    elif hasattr(nnmodel.model, 'layers'): # General case for many HF models
        model_layers_list = nnmodel.model.layers
    else:
        raise AttributeError("Could not find layers in the model structure. Please adapt layer access.")
    
    submodules = [model_layers_list[layer_idx] for layer_idx in layers_indices]
    submodule_names = [f"layer_{layer_idx}" for layer_idx in layers_indices]

    d_model = nnmodel.model.config.hidden_size # Accessing config from the wrapped model
    logger.info(f"d_model={d_model}")

    store_dir_remote = Path(config["activation_store_dir_remote"])
    store_dir_remote.mkdir(parents=True, exist_ok=True)
    
    dataset_name_suffix = config["dataset_name"].split("/")[-1]
    # load_dataset might need token=hf_token if accessing private datasets
    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"], token=hf_token, name=config["dataset_subset"])
    # Select max_samples randomly
    dataset = dataset.shuffle(seed=42) # Ensure reproducibility for the same dataset
    dataset = dataset.select(range(min(config["max_samples"], len(dataset))))

    if config["use_chat_template"]:
        tokenizer.chat_template = config["chat_template"]
        # Create a new column 'text' by applying the chat template
        dataset = dataset.map(
            lambda example: {"text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example['problem']},
                    {"role": "assistant", "content": f"<think>{example['deepseek_reasoning']}</think>{example['deepseek_solution']}"}
                ],
                tokenize=False,
                add_generation_prompt=False)},
            num_proc=1,
            remove_columns=dataset.column_names,
            desc="Applying chat template",
        )

    text_column = CFG["text_column"]
    if config["text_column"] is not None:
        text_column = config["text_column"]

    current_dataset_split_name = config["dataset_split"]
    if text_column != "text":
        current_dataset_split_name = f"{current_dataset_split_name}-col{text_column}"

    logger.info(f"Text column: {text_column}")
    out_dir = store_dir_remote / config["model_name"].split("/")[-1] / dataset_name_suffix / current_dataset_split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Collecting activations to {out_dir}")
    
    # Short delay, as in original script
    time.sleep(2) # Reduced from 10 for potentially faster startup in Modal if not strictly needed

    ActivationCache.collect(
        dataset[text_column],
        submodules,
        submodule_names,
        nnmodel, # Pass the nnsight LanguageModel instance
        out_dir,
        shuffle_shards=False,
        io="out",
        shard_size=10**3, # Consider adjusting based on Modal storage / memory
        batch_size=config["batch_size"],
        context_len=config["context_len"],
        d_model=d_model,
        last_submodule=submodules[-1],
        max_total_tokens=config["max_tokens"],
        store_tokens=config["store_tokens"],
        multiprocessing=not config["disable_multiprocessing"], # Check how this interacts with Modal's parallelism
        ignore_first_n_tokens_per_sample=CFG["ignore_first_n_tokens_per_sample"],
        overwrite=config["overwrite"],
        token_level_replacement=CFG["token_level_replacement"],
    )
    logger.info(f"Activation collection finished. Data saved to {out_dir}")
    return str(out_dir) # Return the path or a status message

@app.local_entrypoint()
def main(
    model: str = DEFAULT_CONFIG["model_name"],
    dataset: str = DEFAULT_CONFIG["dataset_name"],
    layers_str: str = ",".join(map(str, DEFAULT_CONFIG["layers"])) # Layers as comma-separated string
):
    # Example of how to call the Modal function from local entrypoint
    # You can expand this to parse more arguments if needed
    layers = [int(l.strip()) for l in layers_str.split(",")]
    
    logger.info(f"Starting Modal function to collect activations for model: {model}, dataset: {dataset}, layers: {layers}")
    
    # Pass necessary arguments to the modal function
    # Any argument not specified will use the default from DEFAULT_CONFIG or the function's signature
    output_path = collect_activations_on_modal.remote(
        model_name=model,
        dataset_name=dataset,
        layers=layers,
        # wandb_enabled=True, # Optionally enable wandb
        # activation_store_dir_remote="/my_custom_output_path" # Override remote path if needed
    )
    logger.info(f"Modal function finished. Output path: {output_path}")

# To run this from your terminal:
# 1. Ensure Modal CLI is installed and configured.
# 2. Save this script (e.g., as collect_activations_modal.py).
# 3. Deploy: modal deploy collect_activations_modal.py
# 4. Run: modal run collect_activations_modal.py --model "google/gemma-2-2b" --dataset "yahma/alpaca-cleaned" --layers-str "10,15"
# (or use other arguments as defined in main)