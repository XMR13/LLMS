import torch
import tiktoken

from pathlib import Path
from sampling_data import get_dataloader
from new_GPT_Arch import GPT2_124M_Config, GPT_Model, generate_text_simple, generate_text
from helper import text_to_token_ids, token_ids_to_text, visualize_training_results
from torch.optim import AdamW
from tqdm import tqdm

torch.manual_seed(123)
"""
Tahapan dalam melakukan training
1. Evaluate the text ✅
2. Split the training and validation sets ✅
3. Create the training loop
4. Text Generation strategies
"""
#CONFIG AND CONSTANST
dataset_path = Path(r"dataset\the_verdict.txt")
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT_Model(GPT2_124M_Config)
NUM_EPOCHS = 10

with open(dataset_path, mode="r", encoding="utf-8") as f:
    text_data = f.read()

#test the length of the dataset
length_data = len(text_data)
data_tokens = len(tokenizer.encode(text_data))

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

def training_simple(model, train_dataloader, val_dataloaader, 
                    optimizer, device, num_epochs, eval_freq,
                    eval_iter, start_context, tokenizer):
    """
    This function is for training the model using the model and getting 
    the output should be the number of loss
    """
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in tqdm(range(num_epochs)):
        model.train()
        model.to(device)
        #create idx, and loss
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(
                input_batch, target_batch,
                model, device
            )

            #loss backward
            loss.backward()
            #update gradients
            optimizer.step()

            token_seen += input_batch.numel()
            global_step +=1

            #this steps is just an optional steps that could've 
            #been skipped. BUt for education purposes this was added here
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloaader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                #print the results ofthis training
                print(f"Ep {epoch+1} (step: {global_step:06d}):"
                      f"Train Loss : {train_loss:.2f}, "
                      f"Val loss: {val_loss:.2f}."
                      )
        
        generate_and_print_sample(model, tokenizer, 
                                    device, start_context)
            
    return train_losses, val_losses, track_token_seen

def evaluate_model(model,train_loader, val_dataloader,
                   device, eval_iter):
    """
    Merupakan fungsi yang digunakan untuk mengevaluasi model
    biasanya diterapkan setelah melalui proses training
    """
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_dataloader(
            train_loader, model, device,
            num_batches=eval_iter
        )

        val_loss = calculate_loss_dataloader(
            val_dataloader, model,
            device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    A simple function for generating text
    """
    model.eval()
    context_length = model.pos_emb.weight.shape[0]

    #convert the text into token
    encoded = text_to_token_ids(
        start_context, tokenizer=tokenizer
    ).to(device)

    #generate text
    with torch.no_grad():
        output_tokens = generate_text(model, encoded, max_new_tokens=50,
                             context_size=context_length, temperature=1.1, top_k=10)
    #and then you convert it back into text
    decoded_text = token_ids_to_text(output_tokens, tokenizer=tokenizer)
    print(f"text yang dirpint adalah: ")
    print(decoded_text.replace("\n", " "))
    model.train()

def main():
    #testing the dataloader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with torch.no_grad():
        train_loss = calculate_loss_dataloader(train_dataloader, model, device=device)
        val_loss = calculate_loss_dataloader(validation_dataloder, model, device=device)
    print(f"Training loss: {train_loss:.2f}")
    print(f"validation loss: {val_loss:.2f}",'\n')

    #---------------------
    #TRAINING
    #----------------------
    print("TRAINING MODEL:")
    optimizer = AdamW(
        params=model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    train_losses, validation_losses, tokens_seen = training_simple(
        model, train_dataloader=train_dataloader, val_dataloaader=validation_dataloder,
        optimizer=optimizer, device=device, num_epochs=NUM_EPOCHS,
        eval_freq = 5, eval_iter=5, start_context="Every effort moves_you", tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
    visualize_training_results(
        model, train_losses, validation_losses,
        tokens_seen, epochs_tensor
    )




if __name__=="__main__":
    main()