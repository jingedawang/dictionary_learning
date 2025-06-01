import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Configuration ---
DEFAULT_DATA_PATH = "dashboard_data_single_encode_auto_interp.json" # Path to your dashboard data

# --- Helper Functions ---

@st.cache_data # Cache data loading
def load_data(data_path):
    """Loads the dashboard data from a JSON file."""
    path = Path(data_path)
    if not path.exists():
        st.error(f"Error: Data file not found at {data_path}")
        st.stop()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {data_path}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.stop()

def calculate_average_activations(data):
    """Calculates average activations for target latents on Llama and DeepSeek responses."""
    metadata = data.get("metadata", {})
    results_per_item = data.get("results_per_item", [])
    target_latent_indices = metadata.get("target_latent_indices", [])

    if not target_latent_indices or not results_per_item:
        return pd.DataFrame()

    avg_activations = {latent_idx: {"llama": [], "deepseek": []} for latent_idx in target_latent_indices}

    for item in results_per_item:
        # Llama analysis
        if item.get("llama_prompt_analysis"):
            for token_analysis in item["llama_prompt_analysis"]:
                for latent_idx in target_latent_indices:
                    activation = token_analysis["activations_on_target_latents"].get(str(latent_idx), 0.0) # Keys might be strings
                    if activation is not None:
                         avg_activations[latent_idx]["llama"].append(activation)
        
        # DeepSeek analysis
        if item.get("deepseek_prompt_analysis"):
            for token_analysis in item["deepseek_prompt_analysis"]:
                for latent_idx in target_latent_indices:
                    activation = token_analysis["activations_on_target_latents"].get(str(latent_idx), 0.0)
                    if activation is not None:
                        avg_activations[latent_idx]["deepseek"].append(activation)

    # Calculate mean, handling cases with no activations
    summary_data = []
    for latent_idx in target_latent_indices:
        llama_mean = np.mean(avg_activations[latent_idx]["llama"]) if avg_activations[latent_idx]["llama"] else 0
        deepseek_mean = np.mean(avg_activations[latent_idx]["deepseek"]) if avg_activations[latent_idx]["deepseek"] else 0
        summary_data.append({
            "latent_index": latent_idx,
            "llama_avg_activation": llama_mean,
            "deepseek_avg_activation": deepseek_mean
        })
    
    return pd.DataFrame(summary_data)

def plot_activation_comparison(df_activations):
    """Plots a comparison of Llama vs DeepSeek average activations for target latents."""
    if df_activations.empty:
        st.warning("No activation data to plot.")
        return None

    n_latents = len(df_activations)
    index = np.arange(n_latents)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(index - bar_width/2, df_activations["llama_avg_activation"], bar_width, label='Llama')
    rects2 = ax.bar(index + bar_width/2, df_activations["deepseek_avg_activation"], bar_width, label='DeepSeek')

    ax.set_xlabel('Target Latent Index')
    ax.set_ylabel('Average Activation')
    ax.set_title('Average Activation of Target Latents: Llama vs DeepSeek')
    ax.set_xticks(index)
    ax.set_xticklabels(df_activations["latent_index"].astype(str))
    ax.legend()

    fig.tight_layout()
    return fig

def get_activation_color(activation_value, max_abs_activation=1.0, positive_color=(0, 255, 0), negative_color=(255,0,0)):
    """Maps activation value to a background color. Intensity based on value."""
    if activation_value == 0 or max_abs_activation == 0:
        return "transparent"
    
    activation_value = max(0, activation_value - 0.3)
    intensity = min(abs(activation_value) / (max_abs_activation - 0.3), 1.0)
    alpha = intensity * 0.7 # Max alpha 0.7 for readability

    if activation_value > 0:
        r, g, b = positive_color
    else:
        r, g, b = negative_color
        
    return f"rgba({r},{g},{b},{alpha})"

def render_text_with_highlights(text_analysis, selected_latent_idx, tokenizer_decode_func=None):
    """Renders text with tokens highlighted based on activation of the selected latent."""
    if not text_analysis:
        return "<p>No analysis data available for this section.</p>"

    # Find max absolute activation for the selected latent in this text_analysis for normalization
    max_abs_activation = 0
    for token_info in text_analysis:
        activation = token_info["activations_on_target_latents"].get(str(selected_latent_idx), 0.0)
        if activation is not None and abs(activation) > max_abs_activation:
            max_abs_activation = abs(activation)
    if max_abs_activation == 0: # Avoid division by zero if all activations are zero
        max_abs_activation = 1.0

    html_parts = []
    for token_info in text_analysis:
        token_str = token_info["token_str"]
        # Handle special characters for HTML
        token_str_html = token_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>").replace("$", " ").replace("\\", " ")
        
        activation = token_info["activations_on_target_latents"].get(str(selected_latent_idx), 0.0)
        if activation is None: activation = 0.0

        color = get_activation_color(activation, max_abs_activation)
        
        # Add tooltip with activation value
        tooltip = f"Activation: {activation:.4f}"
        html_parts.append(f'<span style="background-color: {color};" title="{tooltip}">{token_str_html}</span>')
    
    return "".join(html_parts)

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("CrossCoder Latent Activation Dashboard")

# --- Data Loading ---
data_file_path = st.sidebar.text_input("Path to dashboard_data.json", DEFAULT_DATA_PATH)
data = load_data(data_file_path)

if not data:
    st.stop()

metadata = data.get("metadata", {})
results_per_item = data.get("results_per_item", [])
target_latent_indices = metadata.get("target_latent_indices", [])

if not results_per_item:
    st.warning("No result items found in the data.")
    st.stop()

if not target_latent_indices:
    st.warning("No target latent indices found in metadata.")
    st.stop()

# --- Section 1: Statistical Analysis ---
st.header("Statistical Analysis of Target Latents")
df_avg_activations = calculate_average_activations(data)

if not df_avg_activations.empty:
    st.dataframe(df_avg_activations.set_index('latent_index'))
    fig = plot_activation_comparison(df_avg_activations)
    if fig:
        st.pyplot(fig)
else:
    st.info("Could not generate statistical analysis. Check data format or content.")

# --- Section 2: Interactive Visualization ---
st.header("Interactive Latent Activation Visualization")

# Selectors
item_ids = [item.get("item_id", f"Item {i}") for i, item in enumerate(results_per_item)]
selected_item_id_display = st.sidebar.selectbox("Select Data Item (by ID or Index):", item_ids)

# Find the actual item based on selection
selected_item_idx = -1
for i, item_disp_id in enumerate(item_ids):
    if item_disp_id == selected_item_id_display:
        selected_item_idx = i
        break

if selected_item_idx == -1:
    st.error("Selected item not found.")
    st.stop()

selected_item_data = results_per_item[selected_item_idx]

# Ensure latent indices are strings for consistent key access if they were loaded as such from JSON keys
# However, target_latent_indices from metadata should be integers. Let's use them as is for selection.
selected_latent_idx = st.sidebar.selectbox("Select Target Latent Index to Visualize:", target_latent_indices)

# --- Display Latent Title and Explanation in Sidebar ---
if selected_latent_idx is not None:
    latent_interpretations = metadata.get("latent_interpretations", {})
    # Ensure selected_latent_idx is treated as a string key if necessary, matching how it might be stored in JSON
    # However, if latent_interpretations keys are integers, convert selected_latent_idx to int.
    # For this example, let's assume keys in latent_interpretations are strings as often happens with JSON.
    interpretation_key = str(selected_latent_idx) 
    selected_latent_info = latent_interpretations.get(interpretation_key)

    if selected_latent_info:
        st.sidebar.markdown("### Latent Interpretation")
        title = selected_latent_info.get("title", "N/A")
        explanation = selected_latent_info.get("explanation", "N/A")
        st.sidebar.markdown(f"**Title:** {title}")
        st.sidebar.markdown(f"**Explanation:** {explanation}")
    else:
        st.sidebar.warning(f"No interpretation data found for latent {selected_latent_idx}.")
# --- End Display Latent Title and Explanation ---

st.subheader(f"Displaying Item: {selected_item_id_display}")
st.markdown(f"**Problem:**")
st.markdown(f"```\n{selected_item_data.get('problem', 'N/A')}\n```")

# Display Llama Response with Highlights
st.markdown(f"**Llama Response (Highlighting for Latent {selected_latent_idx}):**")
llama_analysis = selected_item_data.get("llama_prompt_analysis")
html_llama = render_text_with_highlights(llama_analysis, selected_latent_idx)
st.markdown(html_llama, unsafe_allow_html=True)

# Display DeepSeek Response with Highlights
st.markdown(f"**DeepSeek Response (Highlighting for Latent {selected_latent_idx}):**")
deepseek_analysis = selected_item_data.get("deepseek_prompt_analysis")
html_deepseek = render_text_with_highlights(deepseek_analysis, selected_latent_idx)
st.markdown(html_deepseek, unsafe_allow_html=True)

# --- Metadata Display (Optional) ---
if st.sidebar.checkbox("Show Full Metadata"):
    st.sidebar.subheader("Dataset Metadata")
    st.sidebar.json(metadata)

if st.sidebar.checkbox("Show Selected Item JSON"):
    st.sidebar.subheader(f"JSON for Item: {selected_item_id_display}")
    st.sidebar.json(selected_item_data)


# To run this Streamlit app:
# 1. Save this code as show_dashboard.py
# 2. Make sure dashboard_data.json is in the same directory (or update DEFAULT_DATA_PATH)
# 3. Open your terminal and run: streamlit run show_dashboard.py