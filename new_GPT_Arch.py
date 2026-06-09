import tiktoken
import torch
import torch.nn as nn

"""
Merancang implementasi dari Arsitektur GPT (mengikuti konfigurasi dari GPT-2)
TO DO:
1. Membuat rancangan layer normalisasi 
2. Merancang fukti aktivasi GELU 
3. Membuat feedforward network 
4. Skip connection (residual conncetion)
5. Menggabungkan bagian - bagian tersebut ke arsitektur GPT

"""

GPT2_124M_Config = {
    "vocab_size"    : 50_257, #besar vocab size dari BEP
    "context_length": 1024, #besar context length
    "emb_dim"       : 768, #embedding dimension dari gpt
    "num_heads"     : 12,  #banyak attention heads
    "num_layers"    : 12,  #banyak layer transformer yang akan digunakan
    "drop_rate"     : 0.1, #rate dropout
    "qkv_bias"      : False, #apakah menggunakan kqv bias
}

#set the seed
torch.manual_seed(123)

class GPT_Dummy_model(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        #menetapkan parameters dari model ini
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerblock(cfg)
            for _ in range(cfg["num_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])

        #output untuk decode
        self.out = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=cfg["qkv_bias"]
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)

        #nornalization before the outpyt layer
        x = self.final_norm(x)
        logits = self.out(x)
        return logits
    
class DummyTransformerblock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        """
        Adalah layer normalisasi yang akan diterpakan di akhir
        trasnformer setelah melalui outputnya
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shitft = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        #do the forward method of the self 
        #calcualte the mean and variance
        tensor_mean = x.mean(dim=-1, keepdim=True)
        tensor_variance = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized = (x - tensor_mean)/torch.sqrt(tensor_variance + self.eps)
        return self.scale * normalized + self.shitft

class GELU(nn.Module):
    """Modul neural network untuk fungsi aktivasi GELU"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        hasil = 0.5 * x * (
            1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))
        )
        
        return hasil

class FeedForwardModule(nn.Module):
    """
    Merupakan modul feedforward sederhana yang ada
    pada transformer, gunanya adalah untuk meeplajari representasi 
    dari data dengan cara membesarkan embedding space kemudian 
    mengkompress nya kembali
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn. Linear(in_features=cfg["emb_dim"], out_features=4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(in_features=4 * cfg["emb_dim"], out_features=cfg["emb_dim"])
        )

    def forward(self, X):
        return self.layers(X)
        

def main_exp():
    #menunjukkan hal ini terlebih dahulu
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    #print hasil tokenizer
    token1 = tokenizer.encode(txt1)
    print(f" hasil tokenizer shape  dengan nilai adalah \n {token1}")
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    #stack every element in that said list into a torch stack
    batch = torch.stack(batch, dim=0)
    print(batch)

    #testing_gpt_model
    GPT_MODEL1 = GPT_Dummy_model(GPT2_124M_Config)


def main():
    #berikan contoh saja 
    tensor_contoh = torch.randn(2, 6)
    layer = nn.Sequential(nn.Linear(6,4), nn.ReLU())
    hasil = layer(tensor_contoh)

    #mencari mean dan variance
    ffn = FeedForwardModule(GPT2_124M_Config)
    
    #tensor baru 
    tensor_experimental = torch.randn(2, 3, GPT2_124M_Config["emb_dim"])
    tensor_ffn = ffn(tensor_experimental)
    print(f"sebelum diproses FFN {tensor_experimental.shape}")
    print(f"setelah ffn {tensor_ffn.shape}")






if __name__ == "__main__":
    main()