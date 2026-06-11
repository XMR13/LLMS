import tiktoken
import torch
import torch.nn as nn

from Attention import Multihead_Attention_V2

"""
Merancang implementasi dari Arsitektur GPT (mengikuti konfigurasi dari GPT-2)
TO DO:
1. Membuat rancangan layer normalisasi ✅
2. Merancang fukti aktivasi GELU ✅
3. Membuat feedforward network ✅
4. Skip connection (residual conncetion)✅
5. Membuat transformer blocks✅
6. Menggabungkan bagian - bagian tersebut ke arsitektur GPT✅

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

GPT2_medium = {
    "vocab_size"    : 50_257, #besar vocab size dari BEP
    "context_length": 1024, #besar context length
    "emb_dim"       : 1024, #embedding dimension dari gpt
    "num_heads"     : 16,  #banyak attention heads
    "num_layers"    : 24,  #banyak layer transformer yang akan digunakan
    "drop_rate"     : 0.1, #rate dropout
    "qkv_bias"      : False, #apakah menggunakan kqv bias
}

GPT2_Large = {
    "vocab_size"    : 50_257, #besar vocab size dari BEP
    "context_length": 1024, #besar context length
    "emb_dim"       : 1280, #embedding dimension dari gpt
    "num_heads"     : 20,  #banyak attention heads
    "num_layers"    : 36,  #banyak layer transformer yang akan digunakan
    "drop_rate"     : 0.1, #rate dropout
    "qkv_bias"      : False, #apakah menggunakan kqv bias
}

#set the seed
torch.manual_seed(123)

class GPT_Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        #menetapkan parameters dari model ini
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[Transformerblock(cfg)
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

        #nornalization before the output layer
        x = self.final_norm(x)

        #the output shape of tensor is (batch, embedding_dim, vocab_size)
        logits = self.out(x)
        return logits
    
class Transformerblock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """
        Adalah kelas yang akan digunakan sebagai placeholder untuk blok
        transformer yang ada pada bagian 
        """
        self.attention = Multihead_Attention_V2(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["drop_rate"],
            kqv_bias=cfg["qkv_bias"]
        )
        self.ffn = FeedForwardModule(cfg=cfg)
        self.normalize1 = LayerNorm(cfg["emb_dim"])
        self.normalize2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        """proses step by step dari transformer ini sesuai dengan teori"""
        #pertama inisiasikan terlebih dahulu shortcut connectionya
        #dengan cara menetapkan x di awal sebagai shoritcut
        shortcut = x
        x = self.normalize1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        #ulangi bagian yang atas, namun disini attention digantikan dengan 
        #feedforwadnetwork
        shortcut = x
        x = self.normalize2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

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

class ExampleDeepNeuralNets(nn.Module):
    def __init__(self, layer_sizes, use_shortcut:bool):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), 
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), 
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), 
            GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), 
            GELU()),
            
        ])
        
    def forward(self, x):
        #do a bnackward passes
        for layer in self.layers:
            #first i need to check if it is good enouigh
            layer_output = layer(x)
            #check if we really can use the skip connection
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

def _print_gradients(model, x):
    """
    fungsi sederhana yang akan digunakan untuk mengetes print gradien
    """
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    #kemudian baru mengambil gradiennya
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} memiliki gradient mean sebesar {param.grad.abs().mean().item()}")
            
def generate_text_simple(model, idx,
                         max_new_tokens, context_size):
    """
    Simple text generator
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        #take the last token of the models and find highest probability distirbution
        logits= logits[:, -1, :]
        probability = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probability, dim=-1, keepdim=True)

        #append the token back into the idx
        idx = torch.cat((idx, idx_next), dim=1)

    return idx



def main():
    #menunjukkan hal ini terlebih dahulu
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []

    txt1 = "Every effort moves you"
    #print hasil tokenizer
    token1 = tokenizer.encode(txt1)
    print(f" hasil tokenizer shape  dengan nilai adalah \n {token1}")

    #stack every element in that said list into a torch stack
    batch = torch.tensor(token1).unsqueeze(0)

    #testing_gpt_model
    model = GPT_Model(GPT2_124M_Config)
    output_batch = model(batch)
    
    print(f"\nhasil setelah dari model GPT adalah {output_batch}")
    print("dengan shape", output_batch.shape)

    #generating text
    model.eval()
    token_generate = generate_text_simple(
        model,
        batch,
        max_new_tokens=10,
        context_size=GPT2_124M_Config["context_length"]
    )

    print(f"\n print output: {token_generate}")
    print(f"output_length : {len(token_generate[0])}")
    decoded_teks = tokenizer.decode(token_generate.squeeze(0).tolist())
    print(decoded_teks)
    
    
    

# def main_exp():
#     torch.manual_seed(123)
#     #berikan contoh saja 
#     #siapkan data yang ada
#     contoh_data = torch.randn(2,10, 768)
#     tf_block = Transformerblock(GPT2_124M_Config)
#     output_transformers = tf_block(contoh_data)

#     print(f"sebelum transformer, shape data: {contoh_data.shape}")
#     print(f"setelah transformer, shape data: {output_transformers.shape}")

#     #check apakah sama (tidak sesuai)
#     print(torch.equal(contoh_data, output_transformers))

    





if __name__ == "__main__":
    main()