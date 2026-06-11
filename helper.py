import torch.nn as nn
import torch


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