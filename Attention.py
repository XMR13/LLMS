import torch
import torch.nn as nn


class Multihead_Attention_V1(nn.Module):
    def __init__(self, d_in, d_out, context_length, heads, 
                 dropout, kqv_bias=False):
        super().__init__()

        self.heads = nn.ModuleList(
            [Masked_Attention(d_in, d_out, context_length, dropout,
                              kqv_bias=kqv_bias) 

            for _ in range(heads)
            ]
        )

    def forward(self, x):
        context_vector = [head(x) for head in self.heads]
        return torch.cat(context_vector, dim=-1)
        
class Multihead_Attention_V2(nn.Module):
    #a more compact and efficient implementation of the multihead attention
    def __init__(self, d_in, d_out, context_length, num_heads,
                 dropout, kqv_bias = False):
        
        super().__init__()

        assert (d_out % num_heads == 0), \
            "d_out harus dapat dibagi dengan num_heads"

        #initialize the parameters
        #intialize the multiheads
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = self.d_out // self.num_heads

        #initialize the weights        
        self.W_q = nn.Linear(in_features=d_in, out_features=d_out, bias = kqv_bias)
        self.W_k = nn.Linear(in_features=d_in, out_features=d_out, bias = kqv_bias)
        self.W_v = nn.Linear(in_features=d_in, out_features=d_out, bias = kqv_bias)
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.dropout = nn.Dropout(dropout)
        
        #create the config files
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):

        #1. siapkan query, keys, dan valuesnya
        batch, num_tokens, d_in = x.shape
        query = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        #2. Ubah dimensinya sehingga bisa diproses 
        #dimensi awal (batch, tokens, input) --> single head
        #dimensi akhir (batch, tokens, num_heads, head_dim) --> multihead

        query = query.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)


        #ubah lagi sehingga perkaliannya dapat dilakukan untuk per attention heads
        query = query.transpose(1,2)
        keys =  keys.transpose(1,2)
        values = values.transpose(1,2)

        #3. Calculate the attention score
        att_score = query @ keys.transpose(2,3) #jadi perkalian (tokens, head_dim) x (tokens, num_heads)


        #4. Mask the attention score to become causel attention
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        att_score.masked_fill_(mask_bool, -torch.inf)

        #5. Calculate the attention weights
        att_weights = torch.softmax(
            att_score / keys.shape[-1] ** 0.5, dim=-1
        )
        
        #6. Dropouut
        att_weights = self.dropout(att_weights)

        #7. Jadikan dia sebagai konteks vektor
        context_vector = (att_weights @ values).transpose(1,2) #ubah kembali dari (batch, num_heads, tokens, head_dim) --> (batch, tokens, num_head, head_dim)


        #8. Ubah menjadi output dengan dimensi (batch, d_out, d_out)
        context_vector = context_vector.contiguous().view(
            batch, num_tokens, self.d_out
        )

        #9. Proyeksikan terakhir denganl linear layer

        context_vector = self.out_proj(context_vector)

        return context_vector

def main():
    #contoh self attention with no weights
    torch.manual_seed(123)

    inputs = torch.rand(size=(1024,768))

    #create the query, key, and values first 
    #Khusus untuk input 2

    #test with batch inputs
    batch = torch.stack((inputs, inputs), dim=0)

    print(batch.shape)

    print(f"\n\n Multihead attention  \n\n")

    #persiapkan seednya terlebih dahulu
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape

    multihead_att = Multihead_Attention_V2(
        d_in, d_out=d_in, context_length=context_length,
        num_heads=12, dropout=0.0
    )

    context_vector = multihead_att(batch)
    print(context_vector)
    print(context_vector.shape)

if __name__ == "__main__":
    main()