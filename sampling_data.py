"""Membuat GPT Tokenizer dengan library
    Yang telah teresedia
    
    ⚙ Dibuat oleh Muhammad Rizieq Rizaldi
    2025
"""

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPT_Datasets(Dataset):
    def __init__(self, text, tokenizer, stride, context_size):
        """Meload data ke dalam array dan mengubah bentuknya menjadi sliding window"""
        super().__init__()

        self.text = text
        self.data_ids = []
        self.label_ids = []

        tokenisasi_id = tokenizer.encode(self.text)
        print(f"banyak token yang digunakan untuk kali ini adalah {len(tokenisasi_id)}")

        #lakukan looping untuk mengakses data dengan memanfaatkan sliding window
        #dan bergantung pada context length yang ada

        for i in range(0, len(tokenisasi_id) - context_size, stride):
            #buatkan chunck training dan label
            chunk_train = tokenisasi_id[i: i + context_size]
            chunk_label = tokenisasi_id[i + 1: i + context_size + 1]

            #masukkan ke dalam data dan label
            self.data_ids.append(torch.tensor(chunk_train))
            self.label_ids.append(torch.tensor(chunk_label)) 


    #fungsi untuk mendapatkan panjang dari label
    def __len__(self):
        return len(self.data_ids)
    

    #fungsi untuk meload pasangan data dan label
    def __getitem__(self, idx):
        return self.data_ids[idx], self.label_ids[idx]
    

#merancang data loader untuk digunakan meload data 
def get_dataloader(text, batch_size = 128, context_length=256,
                   stride = 8, shuffle=True, num_workers=0,
                   drop_last=True):
    
    """Fungsi yang berfungsi untuk meload kan data
        yang berasal dari objek dataset, kemudian 
        memprosesnya menjadi sebuah tensor
    """

    #panggil tokenizernya
    tokenizer = tiktoken.get_encoding('gpt2')
    #bangun dataset dari kelas dataset tadi
    dataset = GPT_Datasets(text=text, tokenizer=tokenizer, stride=stride,
                           context_size=context_length)
    
    #memasukkan dataset ke dalam data loader yang ada
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            drop_last=drop_last)
    
    return dataloader

def main():
    #Target, sampai ke tahap berikut:
    #tahapan dalam merancang pipeline tokenizer yang ada ✅
    #load teks yang perlu ✅
    #ubah menjadi token dengan tokneizer yang ada ✅
    #token --> Token ID ✅
    #Rancang dataLoader yang efisien untuk mengakses data tersebut ✅
    #konversi data pasangan tersebut menjadi embedding
    #menambah positional embeeding
    
    #1. Mengakses file yang ingin dijadikan input data
    with open('shakespare.txt', 'r') as f:
        raw_data = f.read()


    #HYPERPARAMETER
    BATCH_SIZE = 8
    STRIDE = 4
    CONTEXT_LENGTH = 4

    #masukkan data tersebut dan proses menjadi dataloader
    dataloader = get_dataloader(raw_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH,
                                stride=STRIDE, shuffle=False)

    
    iter_dataloader = iter(dataloader)

    #batch pertama
    input, label = next(iter_dataloader)
    print('ID token: \n', input)

    print('Label token: \n', label)

    #membuat embedding layer untuk mengubahnya menjadi vektor
    VOCAB_SIZE = 50_257
    OUTPUT_EMBEDDING = 256

    layer_embedding = torch.nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=OUTPUT_EMBEDDING) #memiliki dimensi (50_257 x 256)
    
    print(layer_embedding.weight.shape)

    #applykan pada input embedding
    token_embedding = layer_embedding(input)
    print('tensor shape', token_embedding.shape)

    #positional encoding
    layer_pos_embedding = torch.nn.Embedding(num_embeddings=CONTEXT_LENGTH, embedding_dim=OUTPUT_EMBEDDING)
    pos_encoding = layer_pos_embedding(torch.arange(0,CONTEXT_LENGTH))

    #final embedding
    input_embedding = token_embedding + pos_encoding

    print(f"tensor batch pertama memiliiki size :\n {input_embedding.shape}")

if __name__=="__main__":
    main()