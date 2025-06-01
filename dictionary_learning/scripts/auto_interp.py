import json
import os
from openai import OpenAI # 确保你已经安装了openai库: pip install openai

# 配置您的OpenAI API密钥
# 建议使用环境变量来存储API密钥，而不是直接硬编码在脚本中
# 例如: client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# 或者直接替换 "YOUR_OPENAI_API_KEY"
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

MODEL_NAME = "gpt-4o" # 或者 "gpt-4" 等其他模型

def load_data(file_path):
    """加载JSON数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_prompt_for_latent(latent_id, all_top_tokens_for_latent):
    """为单个latent生成调用LLM的prompt"""
    prompt = f"The following are examples of tokens where latent feature '{latent_id}' is highly active.\n"
    prompt += "Please analyze these tokens and their surrounding context (if provided) to hypothesize what this latent feature might represent. "
    prompt += "Focus on common themes or concepts across these examples.\n\n"

    for i, data_example in enumerate(all_top_tokens_for_latent):
        prompt += f"Example {i+1}:\n"
        if not data_example:
            prompt += "  (No significant activations found for this example for this latent)\n"
            continue
        for item in data_example:
            prompt += f"  - Token: '{item['token']}', Activation: {item['activation']:.4f}\n"
        prompt += "\n"
    
    prompt += "Based on these examples, what does latent feature '{latent_id}' likely represent? Provide a concise explanation."
    return prompt

def get_interpretation_from_llm(prompt):
    """调用LLM API获取解释"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in interpreting latent features from neural networks based on token activations."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error: Could not get interpretation from LLM."

def main():
    data_file_path = 'd:\\Projects\\dictionary_learning\\dashboard_data_single_encode.json'
    dashboard_data = load_data(data_file_path)

    if not dashboard_data:
        print("Failed to load data.")
        return

    metadata = dashboard_data.get('metadata')
    # 根据用户提供的信息，数据在 'results_per_item' 键下
    results_per_item = dashboard_data.get('results_per_item') 

    if not metadata or 'target_latent_indices' not in metadata:
        print("Metadata or target_latent_indices not found in the data.")
        return
    
    if not results_per_item or not isinstance(results_per_item, list):
        print("'results_per_item' field not found or is not a list.")
        return

    target_latent_indices = metadata['target_latent_indices']
    print(f"Target latent indices: {target_latent_indices}\n")

    all_interpretations = {}

    for latent_idx_val in target_latent_indices:
        latent_id_str = str(latent_idx_val) # latent ID是字符串形式的索引
        print(f"--- Interpreting Latent Feature: {latent_id_str} ---")
        
        all_top_tokens_for_this_latent_across_items = []
        
        # 遍历 'results_per_item' 中的每个item (10条数据)
        for item_idx, item_data in enumerate(results_per_item):
            # 优先使用 llama_prompt_analysis, 你可以按需修改为 deepseek_prompt_analysis 或两者都处理
            prompt_analysis_data = item_data.get('llama_prompt_analysis') 
            if not prompt_analysis_data or not isinstance(prompt_analysis_data, list):
                # print(f"  'llama_prompt_analysis' not found or not a list in item {item_idx} for latent {latent_id_str}.")
                all_top_tokens_for_this_latent_across_items.append([])
                continue

            current_item_top_tokens_for_latent = []
            activations_for_current_latent_in_item = []

            # 遍历当前item中的所有token信息
            for token_info in prompt_analysis_data:
                token_str = token_info.get('token_str')
                activations_on_target = token_info.get('activations_on_target_latents')
                
                if token_str is not None and activations_on_target and isinstance(activations_on_target, dict):
                    # 从activations_on_target_latents中获取当前目标latent的激活值
                    activation_value = activations_on_target.get(latent_id_str)
                    if activation_value is not None:
                        activations_for_current_latent_in_item.append({
                            'token': token_str,
                            'activation': activation_value
                        })
            
            if activations_for_current_latent_in_item:
                # 按激活值降序排序并取top N (例如5个)
                sorted_activations = sorted(activations_for_current_latent_in_item, key=lambda x: x['activation'], reverse=True)
                current_item_top_tokens_for_latent = sorted_activations[:5]
            
            all_top_tokens_for_this_latent_across_items.append(current_item_top_tokens_for_latent)
            # print(f"  Item {item_idx+1} for latent {latent_id_str}: Top tokens: {current_item_top_tokens_for_latent}")

        if not any(all_top_tokens_for_this_latent_across_items):
            print(f"No significant activations found for latent {latent_id_str} in any data entry.")
            all_interpretations[latent_id_str] = "No significant activations found."
            continue

        prompt = generate_prompt_for_latent(latent_id_str, all_top_tokens_for_this_latent_across_items)
        print(f"\nGenerated Prompt for Latent {latent_id_str}:\n{prompt}\n") # 取消注释以查看prompt
        exit()
        
        interpretation = get_interpretation_from_llm(prompt)
        print(f"Interpretation for Latent {latent_id_str}:\n{interpretation}\n")
        all_interpretations[latent_id_str] = interpretation

    print("\n--- All Latent Interpretations ---")
    for latent_id, interp_text in all_interpretations.items():
        print(f"Latent {latent_id}: {interp_text}")

    # 你可以将结果保存到文件
    # output_file_path = 'latent_interpretations.json'
    # with open(output_file_path, 'w', encoding='utf-8') as outfile:
    #     json.dump(all_interpretations, outfile, indent=2, ensure_ascii=False)
    # print(f"\nInterpretations saved to {output_file_path}")

if __name__ == '__main__':
    main()