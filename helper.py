import torch.nn as nn
import torch
import tiktoken
import matplotlib.pyplot as plt
import urllib.request

from matplotlib.ticker import MaxNLocator

def calculate_params(model:nn.Module):
    """adalah fungsi helper sederhan untuk menghitung banyak parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def calculate_size(num_params) -> float:
    """mehgiutng size dari model apabila semua menggunakan 4 bytes"""
    total_bytes = num_params * 4

    if total_bytes > 1e9:
        model_in_gb = total_bytes / (1024 * 1024 * 1024)
        print(f"besar model adalah {model_in_gb:.2f}GB.")
    
    else:
        model_in_mb = total_bytes / (1024 * 1024)
        print(f"besar model adalah {model_in_mb:.2f}MB.")

def text_to_token_ids(text, tokenizer):
    """convert text into tokens"""
    token = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(token).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(tokens, tokenizer):
    """
    convert the token back into text
    """
    text_clean = tokens.squeeze(0)
    text_decode = tokenizer.decode(text_clean.tolist())
    return text_decode

def visualize_training_results(model, training_losses, validation_losses,
                               tokens_seen, epochs_seen):
    """
    Fungsi helper sederhana yang akan digunakan untuk memvisualisasikan 
    nilai antara
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, training_losses, label="Training loss")
    ax1.plot(
    epochs_seen, validation_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, training_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

#-------------------
# MODEL UTILITY
#-------------------
def save_model(model, name_model, optimizer=None):
    """
    simple function to save the model
    """
    final_model_name : str = name_model + ".pth"
    if optimizer is not None:
        torch.save({
            "model_state_dict" :model.state_dict(), 
            "optimizer_state_dict":optimizer.state_dict(),
            },
            final_model_name    
        )
    else:
        torch.save(
            model.state_dict(),
            final_model_name
        )

def load_model(model, model_path, device):
    """
    simple function to load the model
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def download_gpt_function():
    url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
    )
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

import numpy as np
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attention.W_q.weight = assign(
        gpt.trf_blocks[b].attention.W_q.weight, q_w.T)
        gpt.trf_blocks[b].attention.W_k.weight = assign(
        gpt.trf_blocks[b].attention.W_k.weight, k_w.T)
        gpt.trf_blocks[b].attention.W_v.weight = assign(
        gpt.trf_blocks[b].attention.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attention.W_q.bias = assign(
        gpt.trf_blocks[b].attention.W_q.bias, q_b)
        gpt.trf_blocks[b].attention.W_k.bias = assign(
        gpt.trf_blocks[b].attention.W_k.bias, k_b)
        gpt.trf_blocks[b].attention.W_v.bias = assign(
        gpt.trf_blocks[b].attention.W_v.bias, v_b)

        gpt.trf_blocks[b].attention.out_proj.weight = assign(
        gpt.trf_blocks[b].attention.out_proj.weight,
        params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attention.out_proj.bias = assign(
        gpt.trf_blocks[b].attention.out_proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ffn.layers[0].weight = assign(
        gpt.trf_blocks[b].ffn.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ffn.layers[0].bias = assign(
        gpt.trf_blocks[b].ffn.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ffn.layers[2].weight = assign(
        gpt.trf_blocks[b].ffn.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ffn.layers[2].bias = assign(
        gpt.trf_blocks[b].ffn.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].normalize1.scale = assign(
        gpt.trf_blocks[b].normalize1.scale,
        params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].normalize1.shift = assign(
        gpt.trf_blocks[b].normalize1.shift,
        params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].normalize2.scale = assign(
        gpt.trf_blocks[b].normalize2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].normalize2.shift = assign(
        gpt.trf_blocks[b].normalize2.shift,
        params["blocks"][b]["ln_2"]["b"])
        
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out.weight = assign(gpt.out.weight, params["wte"])
