import torch
import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer
from nnsight import LanguageModel # 用于获取模型激活

# 假设 dictionary_learning 在 PYTHONPATH 或可访问
# from dictionary_learning.cache import PairedActivationCache, ActivationCache # 不再需要从缓存加载
from dictionary_learning.dictionary import CrossCoder, BatchTopKCrossCoder

import modal
import sys

# --- Modal 配置 ---
project_root_local = "d:/Projects/dictionary_learning"
remote_project_root = "/root/dictionary_learning"

# 默认配置 (可以被参数覆盖)
DEFAULT_REMOTE_ACTIVATION_STORE_DIR = "/data/activation_store_2" # 虽然不再直接使用，但保留以防其他部分间接依赖路径结构
DEFAULT_BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
DEFAULT_BASE_MODEL_NAME = "Meta-Llama-3.1-8B" # 用于路径和模型加载
DEFAULT_INSTRUCT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_INSTRUCT_MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B" # 用于路径和模型加载
DEFAULT_DATASET_PATH = f"{remote_project_root}/dual_model_math500_dataset.json" # 本地JSON文件路径
DEFAULT_LAYER = 12 # 第13层，索引为12
DEFAULT_CROSSCODER_MODEL_PATH = "/data/checkpoints/openthoughts-110k-50000/model_final.pt" # Modal卷上的示例路径
DEFAULT_OUTPUT_FILE_PATH = "/data/analysis/openthoughts-110k-50000/v3/dashboard_data_single_encode_both_models.txt"
DEFAULT_MODEL_TYPE = "crosscoder" # "crosscoder" 或 "batchtopkcrosscoder"
DEFAULT_MAX_LENGTH = 512 # Tokenizer的最大长度

# nnsight的tracer参数
tracer_kwargs = {"scan": False, "validate": False}

def get_target_latent_indices(crosscoder_model, top_k=10, bottom_k=10):
    """计算相对解码器范数，并返回top_k和bottom_k的潜在激活的索引，按相对范数降序排列。"""
    with torch.no_grad():
        # 假设 crosscoder_model.decoder.weight 的形状是 (num_crosscoder_layers, dict_size, activation_dim)
        # CrossCoder的decoder.weight[0] 对应 base_model 的重构, decoder.weight[1] 对应 instruct_model 的重构
        if crosscoder_model.decoder.weight.shape[0] < 2:
            raise ValueError("CrossCoder模型必须至少有2个解码器层用于基模型和推理模型的范数计算。")

        base_norms = torch.norm(crosscoder_model.decoder.weight[0], p=2, dim=1) # 对每个latent的activation_dim维向量求范数
        reasoning_norms = torch.norm(crosscoder_model.decoder.weight[1], p=2, dim=1)

        if base_norms.shape != reasoning_norms.shape:
            raise ValueError(f"基模型范数 {base_norms.shape} 和推理模型范数 {reasoning_norms.shape} 形状不匹配")
        if len(base_norms.shape) != 1:
            raise ValueError(f"范数不是1D张量。基模型范数形状: {base_norms.shape}, 推理模型范数形状: {reasoning_norms.shape}")

        relative_norms = reasoning_norms / (base_norms + reasoning_norms + 1e-9) # 为稳定性添加epsilon
        
        # 获取相对范数最大的top_k个潜在激活
        top_k_indices = torch.topk(relative_norms, top_k).indices

        # 获取相对范数最小的bottom_k个潜在激活
        # torch.topk with largest=False gives smallest values
        bottom_k_indices = torch.topk(relative_norms, bottom_k, largest=False).indices

        # 合并top_k和bottom_k的索引，并去重
        combined_indices = torch.cat([top_k_indices, bottom_k_indices])
        final_indices = torch.unique(combined_indices)

        # 按照相对范数降序排序
        final_indices = final_indices[torch.argsort(relative_norms[final_indices], descending=True)]

    return final_indices, relative_norms

def load_models_and_tokenizer(crosscoder_model_path, model_type, base_model_hf_id, instruct_model_hf_id, device):
    """加载CrossCoder模型、底层的LLMs（用于提取激活）和分词器。"""
    print(f"正在加载 CrossCoder 模型 ({model_type}) 从: {crosscoder_model_path}")
    if model_type == "batchtopkcrosscoder":
        crosscoder_model = BatchTopKCrossCoder.from_pretrained(crosscoder_model_path, device=device, dtype=torch.float32)
    elif model_type == "crosscoder":
        crosscoder_model = CrossCoder.from_pretrained(crosscoder_model_path, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    crosscoder_model.eval()

    print(f"正在加载基础LLM: {base_model_hf_id}")
    base_llm = LanguageModel(base_model_hf_id, trust_remote_code=True)
    print(f"正在加载指令LLM: {instruct_model_hf_id}")
    instruct_llm = LanguageModel(instruct_model_hf_id, trust_remote_code=True)

    # 通常我们使用其中一个模型的tokenizer，或者一个通用的。这里用基础模型的。
    print(f"正在加载分词器: {base_model_hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"已将分词器的 pad_token 设置为 eos_token: {tokenizer.eos_token}")
    tokenizer.padding_side = 'right' # 通常对于生成激活，右填充更方便

    return crosscoder_model, base_llm, instruct_llm, tokenizer

def get_activations_from_llm(prompt_text, llm_model, tokenizer, layer_idx, device, max_length=DEFAULT_MAX_LENGTH):
    """辅助函数：为给定提示词从指定LLM的特定层获取激活。"""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").to(device)
    
    with llm_model.trace(inputs['input_ids'], **tracer_kwargs) as runner:
        activations = llm_model.model.layers[layer_idx].output.save()
    
    # 确保激活在正确的设备上，并且是float32类型
    # activation是一个元组，包含 (tensor,)
    act_val = activations[0].squeeze(0).to(device).float() # 移除batch维度
    
    # 根据 attention_mask 过滤掉 padding tokens 的激活
    # inputs['attention_mask'] 形状是 (batch_size, sequence_length)
    attention_mask_squeezed = inputs['attention_mask'].squeeze(0) # 形状 (sequence_length,)
    actual_tokens_mask = attention_mask_squeezed == 1
    
    return act_val[actual_tokens_mask] # 返回 (num_actual_tokens, hidden_size)


def analyze_prompt_activations(
    prompt_text: str,
    is_base_model: bool,
    llm: LanguageModel,
    crosscoder_model: CrossCoder,
    tokenizer: AutoTokenizer,
    target_latent_indices: torch.Tensor,
    layer_idx: int,
    device: str,
    max_length: int
):
    """
    为给定的提示词，从base_llm和instruct_llm获取激活，
    然后通过CrossCoder得到潜在激活，并分析目标潜在激活。
    """
    results_for_prompt = []

    # 1. 从LLM获取激活
    if is_base_model:
        model_acts_base = get_activations_from_llm(prompt_text, llm, tokenizer, layer_idx, device, max_length)
        # Set chat acts as zeros
        model_acts_instruct = torch.zeros_like(model_acts_base)
    else:
        model_acts_instruct = get_activations_from_llm(prompt_text, llm, tokenizer, layer_idx, device, max_length)
        # Set base acts as zeros
        model_acts_base = torch.zeros_like(model_acts_instruct)

    # 2. 准备CrossCoder的输入
    # CrossCoder期望输入形状为 (batch_size_tokens, num_models, activation_dim)
    stacked_acts = torch.stack([model_acts_base, model_acts_instruct], dim=1).to(device).float()

    # 3. 通过CrossCoder获取潜在激活
    with torch.no_grad():
        # crosscoder_model.encode 的输出形状是 (num_tokens, dict_size)
        latent_acts_sum, latents_acts_seperate = crosscoder_model.encode(stacked_acts, return_no_sum=True)
        # latent_acts_seperate 形状是 (num_tokens, num_models, activation_dim)
        if is_base_model:
            latent_acts_all = latents_acts_seperate[:, 0]
        else:
            latent_acts_all = latents_acts_seperate[:, 1]
        # latent_acts_all 形状是 (num_tokens, activation_dim)

    # 4. 分析目标潜在激活并记录结果
    # 获取对应prompt的token IDs (未被padding的部分)
    token_ids_for_prompt = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)['input_ids'].squeeze().tolist()

    for token_idx in range(len(token_ids_for_prompt)):
        token_id = token_ids_for_prompt[token_idx]
        token_str = tokenizer.decode([token_id])
        
        activations_on_targets = {}
        for latent_idx_tensor in target_latent_indices:
            latent_idx = latent_idx_tensor.item()
            activation_value = latent_acts_all[token_idx, latent_idx].item()
            activations_on_targets[latent_idx] = activation_value
        
        results_for_prompt.append({
            "token_idx_in_prompt": token_idx,
            "token_id": token_id,
            "token_str": token_str,
            "activations_on_target_latents": activations_on_targets
        })
        
    return results_for_prompt


def process(
    crosscoder_model_path: str,
    model_type: str,
    # activation_store_dir: str, # 不再使用
    base_model_name_or_hf_id: str, 
    instruct_model_name_or_hf_id: str,
    dataset_path: str, # JSON文件路径
    layer: int, # LLM的层索引 (0-indexed)
    relative_norm_threshold: float,
    # num_top_sequences: int, # 不再按top sequences, 而是分析整个prompt
    output_file: str,
    max_length_tokenizer: int,
    device_str: str = None
):
    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
    print(f"使用设备: {device}")

    # 1. 加载 CrossCoder 模型, LLMs 和分词器
    crosscoder_model, base_llm, instruct_llm, tokenizer = load_models_and_tokenizer(
        crosscoder_model_path, model_type, base_model_name_or_hf_id, instruct_model_name_or_hf_id, device
    )

    # 2. 获取目标潜在激活索引
    target_indices, all_relative_norms = get_target_latent_indices(
        crosscoder_model
    )
    if target_indices.numel() == 0:
        print(f"没有找到相对范数 > {relative_norm_threshold} 的潜在激活。正在退出。")
        return
    print(f"找到 {target_indices.numel()} 个目标潜在激活，其相对范数 > {relative_norm_threshold}。")
    # print(f"目标潜在激活索引: {target_indices.tolist()}")

    # 3. 加载JSON数据集
    print(f"正在从 {dataset_path} 加载数据集...")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据集文件 {dataset_path} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析数据集文件 {dataset_path}失败。")
        return
    
    if not isinstance(dataset, list) or not dataset:
        print("错误: 数据集格式不正确或为空。")
        return

    print(f"数据集加载成功，包含 {len(dataset)} 个条目。")

    # 4. 对每个数据条目进行分析
    full_dashboard_results = []
    print(f"开始处理 {len(dataset)} 个数据条目...")

    for i, item in enumerate(dataset):
        print(f"\n--- 正在处理条目 {i+1}/{len(dataset)} (ID: {item.get('id', 'N/A')}) ---")
        prompt_used = item.get("prompt_used", "")
        llama_response_text = item.get("llama_response", "")
        deepseek_response_text = item.get("deepseek_response", "")

        llama_full_prompt = prompt_used + llama_response_text
        deepseek_full_prompt = prompt_used + deepseek_response_text
        
        item_results = {
            "item_id": item.get('id'),
            "problem": item.get('problem'),
            "llama_response": llama_response_text,
            "deepseek_response": deepseek_response_text,
            "llama_prompt_analysis": None,
            "deepseek_prompt_analysis": None
        }

        print(f"  正在分析Llama完整提示 (长度: {len(llama_full_prompt)} chars)...")
        item_results["llama_prompt_analysis"] = analyze_prompt_activations(
            prompt_text=llama_full_prompt,
            is_base_model=True,
            llm=base_llm,
            crosscoder_model=crosscoder_model,
            tokenizer=tokenizer,
            target_latent_indices=target_indices,
            layer_idx=layer,
            device=device,
            max_length=max_length_tokenizer
        )
        print(f"  Llama完整提示分析完成。")

        print(f"  正在分析Deepseek完整提示 (长度: {len(deepseek_full_prompt)} chars)...")
        item_results["deepseek_prompt_analysis"] = analyze_prompt_activations(
            prompt_text=deepseek_full_prompt,
            is_base_model=False,
            llm=instruct_llm,
            crosscoder_model=crosscoder_model,
            tokenizer=tokenizer,
            target_latent_indices=target_indices,
            layer_idx=layer,
            device=device,
            max_length=max_length_tokenizer
        )
        print(f"  Deepseek完整提示分析完成。")
        
        full_dashboard_results.append(item_results)

    # 5. 保存或打印结果
    # 为了方便阅读，我们将结果保存为JSON文件，而不是文本文件。
    output_json_file = Path(output_file).with_suffix(".json")
    print(f"\n正在将详细结果保存到 {output_json_file}...")
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "crosscoder_model_path": crosscoder_model_path,
                "base_model_hf_id": base_model_name_or_hf_id,
                "instruct_model_hf_id": instruct_model_name_or_hf_id,
                "dataset_path": dataset_path,
                "llm_layer_idx": layer,
                "relative_norm_threshold": relative_norm_threshold,
                "num_target_latents": target_indices.numel(),
                "target_latent_indices": target_indices.tolist(),
                "target_latents": {idx: all_relative_norms[idx].item() for idx in target_indices.tolist()},
                "all_relative_norms": all_relative_norms.tolist() # 保存所有相对范数以供参考
            },
            "results_per_item": full_dashboard_results
        }, f, ensure_ascii=False, indent=2)
    print(f"仪表盘结果已保存到 {output_json_file}")

    # （可选）可以生成一个简化的文本摘要文件
    output_summary_file = Path(output_file) # 原本的txt文件用于摘要
    print(f"正在生成摘要到 {output_summary_file}...")
    with open(output_summary_file, "w", encoding="utf-8") as f:
        f.write(f"CrossCoder Latent Dashboard Summary\n")
        f.write(f"CrossCoder Model: {crosscoder_model_path}\n")
        f.write(f"Base LLM: {base_model_name_or_hf_id}, Instruct LLM: {instruct_model_name_or_hf_id}, Layer: {layer}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Relative Norm Threshold for Latents: > {relative_norm_threshold}\n")
        f.write(f"Number of Target Latents: {target_indices.numel()}\n")
        f.write(f"Target Latent indices with relative norms: { {idx: all_relative_norms[idx].item() for idx in target_indices.tolist()} } \n\n")

        for item_data in full_dashboard_results:
            f.write(f"--- Item ID: {item_data['item_id']} ---\n")
            f.write(f"Problem: {item_data['problem'][:200]}...\n\n") # 截断问题以保持摘要简洁

            for prompt_type in ["llama_prompt_analysis", "deepseek_prompt_analysis"]:
                f.write(f"  Analysis for {prompt_type.replace('_analysis', '')}:\n")
                analysis_results = item_data[prompt_type]
                if not analysis_results:
                    f.write("    No tokens analyzed for this prompt.\n")
                    continue

                # 仅展示每个目标潜在激活上激活值最高的几个token的摘要信息
                for latent_idx_tensor in target_indices:
                    latent_idx = latent_idx_tensor.item()
                    f.write(f"    Latent {latent_idx} (Rel Norm: {all_relative_norms[latent_idx]:.4f}):\n")
                    
                    # 收集此潜在激活的所有激活值和对应token
                    token_activations_for_latent = []
                    for token_info in analysis_results:
                        token_activations_for_latent.append(
                            (token_info["activations_on_target_latents"].get(latent_idx, 0.0), 
                             token_info["token_str"],
                             token_info["token_idx_in_prompt"])
                        )
                    
                    # 按激活值降序排序
                    token_activations_for_latent.sort(key=lambda x: x[0], reverse=True)
                    
                    # 显示激活值最高的 N=3 个token
                    for act_val, token_s, token_i in token_activations_for_latent[:3]:
                        f.write(f"      Token @{token_i} \"{token_s}\": Activation {act_val:.4f}\n")
                    if not token_activations_for_latent:
                         f.write(f"      No significant activations found for this latent.\n")
                f.write("\n")
            f.write("\n")
    print(f"摘要已保存到 {output_summary_file}")


# 将dual_model_math500_dataset.json添加到镜像中
# 确保此文件与您的 crosscoder_dashboard_for_math500.py 在同一目录或可访问的相对路径
# 或者，如果您在本地运行此脚本以提交到Modal，则将其放在正确的位置。
# 在这个例子中，我假设它在项目根目录。


image = (
    modal.Image.debian_slim(python_version="3.10") # 确保Python版本与依赖兼容
    .pip_install_from_requirements(f"{project_root_local}/requirements.txt")
    # 或者如果它是通过 requirements.txt 安装的 git repo，那也没问题
    .apt_install("git") # nnsight可能需要git
    .pip_install("nnsight", "einops") # 显式安装nnsight和einops
    .add_local_dir(project_root_local, remote_path=remote_project_root)
)

app = modal.App("crosscoder-json-dashboard")

@app.function(
    image=image,
    gpu="H100", # 例如 A10G, T4, 或 H100 (如果需要且可用)
    secrets=[modal.Secret.from_name("huggingface-secret-jinge"), modal.Secret.from_name("wandb-secret")], # HuggingFace token, wandb可选
    volumes={"/data": modal.Volume.from_name("crosscoder")},
    timeout=3600 * 2, # 2小时超时，根据需要调整
    # container_idle_timeout=300 # 容器空闲超时（秒）
    # concurrency_limit=1 # 如果模型加载很重，限制并发
    # memory=... # 如果需要，指定内存 e.g., 8192 for 8GB
)
def process_wrapper(**kwargs):
    process(**kwargs)

@app.local_entrypoint()
def main(
    crosscoder_model_path: str = DEFAULT_CROSSCODER_MODEL_PATH,
    model_type: str = DEFAULT_MODEL_TYPE,
    # activation_store_dir: str = DEFAULT_REMOTE_ACTIVATION_STORE_DIR, # 不再需要
    base_model_name_or_hf_id: str = DEFAULT_BASE_MODEL_ID,
    instruct_model_name_or_hf_id: str = DEFAULT_INSTRUCT_MODEL_ID,
    dataset_path: str = DEFAULT_DATASET_PATH, # 本地路径，在wrapper中会被替换
    layer: int = DEFAULT_LAYER,
    relative_norm_threshold: float = 0.9,
    # num_top_sequences: int = 5, # 不再需要
    output_file: str = DEFAULT_OUTPUT_FILE_PATH,
    max_length_tokenizer: int = DEFAULT_MAX_LENGTH,
    device: str = None
):
    print(f"运行 CrossCoder (JSON输入) 仪表盘生成，参数如下:")
    print(f"  CrossCoder模型路径: {crosscoder_model_path}")
    print(f"  模型类型: {model_type}")
    print(f"  基础LLM: {base_model_name_or_hf_id}")
    print(f"  指令LLM: {instruct_model_name_or_hf_id}")
    print(f"  数据集路径 (本地，Modal将使用镜像内路径): {dataset_path}")
    print(f"  LLM层索引: {layer}")
    print(f"  相对范数阈值: {relative_norm_threshold}")
    print(f"  Tokenizer最大长度: {max_length_tokenizer}")
    print(f"  输出文件: {output_file}")
    print(f"  设备: {device if device else '自动选择'}")

    process_wrapper.remote(
        crosscoder_model_path=crosscoder_model_path,
        model_type=model_type,
        base_model_name_or_hf_id=base_model_name_or_hf_id,
        instruct_model_name_or_hf_id=instruct_model_name_or_hf_id,
        dataset_path=dataset_path, # 这个路径参数主要用于本地显示，在 remote 中会被覆盖
        layer=layer,
        relative_norm_threshold=relative_norm_threshold,
        output_file=output_file,
        max_length_tokenizer=max_length_tokenizer,
        device_str=device
    )

if __name__ == "__main__":
    # 这使得你可以使用 `python crosscoder_dashboard_for_math500.py --help` 来查看参数
    # 并通过命令行参数运行，例如 `python crosscoder_dashboard_for_math500.py --layer 10`
    # modal.runner.deploy_stub(stub) # 如果你想部署为一个持久应用
    # 对于直接运行脚本，不需要上面这行
    # 当你从本地运行 `modal run crosscoder_dashboard_for_math500.py` 时,
    # `app.local_entrypoint()` 会被调用。
    pass