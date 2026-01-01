from transformers import BartTokenizer, BartModel
import torch
from torch import nn
from tqdm import tqdm

messages = [
    "We have release a new product, do you want to buy it?", 
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you come to the party?",
]
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartModel.from_pretrained("facebook/bart-base")

def convert_to_embeddings(messages):
    embeddings_list = []
    for message in tqdm(messages):
        # Here we just want to pass in a single message, but it needs to be a Python list. So I need to wrap this into square brackets to turn this single message into a list.
        out = tokenizer([message], 
                        padding=True,
                        max_length=512, 
                        truncation=True, 
                        return_tensors="pt")

        with torch.no_grad():
            bart_model.eval()

            pred = bart_model(
                input_ids=out["input_ids"], 
                attention_mask=out["attention_mask"]
            )
            embeddings = pred.last_hidden_state.mean(dim=1)\
                .reshape((-1))
            
            # print(embeddings.shape)
            # print(embeddings)

            embeddings_list.append(embeddings)
    return torch.stack(embeddings_list)


X = convert_to_embeddings(messages)
print(X.shape)