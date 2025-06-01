# debugging flag for use in other scripts
DEBUG = False

MODEL_CONFIGS = {
    "Qwen/Qwen2.5-1.5B": {
        "ignore_first_n_tokens_per_sample": 21,
        "text_column": "text_qwen2_5",
        "attn_implementation": None,
        "token_level_replacement": None,
        "token_level_replacement": None,
    },
    "google/gemma-2-2b": {
        "ignore_first_n_tokens_per_sample": 0,
        "text_column": "text",
        "attn_implementation": "eager",
        "token_level_replacement": None,
    },
    "meta-llama/Meta-Llama-3.1-8B": {
        "ignore_first_n_tokens_per_sample": 25,
        "text_column": "text_llama3",
        "attn_implementation": None,
        "token_level_replacement": None,
        "token_level_replacement": None,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "ignore_first_n_tokens_per_sample": 0,
        "text_column": "text",
        "attn_implementation": None,
        "token_level_replacement": None,
    }
}
MODEL_CONFIGS["google/gemma-2-2b-it"] = MODEL_CONFIGS["google/gemma-2-2b"]
MODEL_CONFIGS["Qwen/Qwen2.5-1.5B-Instruct"] = MODEL_CONFIGS["Qwen/Qwen2.5-1.5B"]
MODEL_CONFIGS["meta-llama/Meta-Llama-3.1-8B-Instruct"] = MODEL_CONFIGS[
    "meta-llama/Meta-Llama-3.1-8B"
]
MODEL_CONFIGS["meta-llama/Meta-Llama-3.1-8B"]["token_level_replacement"] = {
    128006: 1432,
    128009: 827,
    128007: 827,
}  # Llama 3.1 Base doesn't deal well with template tokens

MODEL_CONFIGS["unsloth/Meta-Llama-3.1-8B"] = MODEL_CONFIGS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"].copy()
MODEL_CONFIGS["unsloth/Meta-Llama-3.1-8B"]["token_level_replacement"] = {
    128011: 827,
    128012: 827,
    128013: 827,
    128014: 827,
}  # Llama 3.1 Base doesn't deal well with template tokens