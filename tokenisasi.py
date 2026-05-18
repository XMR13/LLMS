import urllib.request
import os
import numpy as np
import re
import tiktoken


url_text_data = (
    "https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/refs/heads/"
    "master/shakespeare.txt"
)
nama_file = "shakespare.txt"

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encodde(self, text):
        """mengencode teks menjadi token serta token id
        
        Args:
            teks: teks yang ingin diencode"""
        
        #cara encoding
        preprocessed_data = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed_data = [
            item.strip() for item in preprocessed_data if item.strip()
        ]
        
        preprocessed_data = [item if item in self.str_to_int
                             else "<|unkwn|>" for item in preprocessed_data]
        

        hasil_encoding_id = [self.str_to_int[s] for s in preprocessed_data]
        return hasil_encoding_id
    
    def decode(self, ids):
        """mengubah kembali dari encoding id menjadi teks
        
        Args:
            ids: id encoding
        """
        
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def encoding(text):
    """Merupakan sebuah fungsi yang berfungsi untuk mengencode text 
        dan mengubahnya menjadi token
    
    Args:
        text: teks yang ingin di encode
        """
    #menentukan repr yang ingin dijadikan hasil encoding
    hasil_encode = re.split(r'([.,!?;:"()\']|--|\s)', text)
    hasil_encode = [item.strip() for item in hasil_encode if item.strip()]
    return hasil_encode

def tokenisasi(hasil_encoding):
    """Merupakan sebuah fugsi yang akan mengubah semua token yagn dikestrak
        menjadi suatu token ID yang merepresentasikan angka dengan token tersebut

    Args:
        hasil_encoding (list): hasil encoding yang akan diubah menjadi token

    """

    hasil_encode = sorted(set(hasil_encoding))
    hasil_encode.extend(["<|endoftext|>", "<|unkwn|>"])
    #melakukan mapping dari awal integer dan hasil encoding
    #membuat vocabulary
    vocab = {token:integer  for integer,token in enumerate(hasil_encode)}
    return vocab


def main():
    #menggunakan byte pair encoding

    #mengambil teks dari sumber shakespare
    with open("shakespare.txt", "r", encoding='utf-8') as f:
        data_raw = f.read()

    print(f"panjang dari teks ini adalah {len(data_raw)}")

    tokenizer = tiktoken.get_encoding('gpt2')

    #contoh teks
    teks = (
        "Hello there, my beautiful lady, may I have the privellege "
        "to dance with one of you?"
    )
    
    integer_id = tokenizer.encode(teks, allowed_special={"<|endoftext|>"})
    print(integer_id)
    print(type(integer_id))

    #mendecode kembali
    string_org = tokenizer.decode(integer_id)
    print(string_org)

    # #contoh lain
    teks2 = "Akwirw ier"
    integer_id2 = tokenizer.encode(teks2, allowed_special={"<|endoftext|>"})

    for i in integer_id2:
        hasil_decodenya = tokenizer.decode([i])
        print(f"encode: {i} ")
        print(f"decode: {hasil_decodenya}\n")

    #decode back
    string_org2 = tokenizer.decode(integer_id2)
    print(string_org2)

if __name__ == '__main__':
    main()