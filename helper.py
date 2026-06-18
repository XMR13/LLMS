import torch.nn as nn
import torch
import tiktoken

from new_GPT_Arch import generate_text_simple

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
