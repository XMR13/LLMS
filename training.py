import torch
import tiktoken

from pathlib import Path
from sampling_data import get_dataloader
from new_GPT_Arch import GPT2_124M_Config, GPT_Model
torch.manual_seed(123)
"""
Tahapan dalam melakukan training
1. Evaluate the text ✅
2. Split the training and validation sets ✅
3. Create the training loop
4. Text Generation strategies
"""

dataset_path = Path(r"dataset\the_verdict.txt")
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT_Model(GPT2_124M_Config)

with open(dataset_path, mode="r", encoding="utf-8") as f:
    text_data = f.read()

#test the length of the dataset
length_data = len(text_data)
data_tokens = len(tokenizer.encode(text_data))
print("banyak data:", length_data)
print("banyak token:", data_tokens)

train_ratio = 0.9
split_idx = int(train_ratio * length_data)
train_data = text_data[:split_idx]
validation_data =text_data[split_idx:]

print(len(train_data))
print(len(validation_data))

#create dataloader
training_context_length = 256
train_dataloader = get_dataloader(
    train_data, 
    batch_size=2,
    context_length=training_context_length,
    stride=training_context_length,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

validation_dataloder = get_dataloader(
    validation_data,
    batch_size=2,
    context_length=training_context_length,
    stride=training_context_length,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

print("train dataloder")
for x, y in train_dataloader:
    print(x.shape, y.shape)

def calculate_loss_batch(input_batch, target_batch, model, device):
    """Calculate the loss of a specific batch"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )

    return loss

def calculate_loss_dataloader(dataloader, model, device, num_batches=None):
    """
    function to calculate the loss of a single pytorch dataloader
    usefull for knowing the loss values of an entire dataset
    """
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch,
                                        model, device)
            total_loss += loss.item()
        else:
            break

    #this mark the total loss of a specifc batches
    return total_loss / num_batches

#testing the dataloader
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
with torch.no_grad():
    train_loss = calculate_loss_dataloader(train_dataloader, model, device=device)
    val_loss = calculate_loss_dataloader(validation_dataloder, model, device=device)
print(f"Training loss: {train_loss:.2f}")
print(f"validation loss: {val_loss:.2f}")