import torch.nn as nn
import torch
import tiktoken
import matplotlib.pyplot as plt

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
