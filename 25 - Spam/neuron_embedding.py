import torch 
from torch import nn
from transformers import BartTokenizer, BartModel
from tqdm import tqdm #for progress bar
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartModel.from_pretrained("facebook/bart-base")

# we got this function here convert to embeddings where we can just throw in a Python list of messages. And it then just gets converted into embeddings.
def convert_to_embeddings(messages):
    embeddings_list = []
    for message in tqdm(messages):
        out = tokenizer([message], padding=True, max_length=512, truncation=True, return_tensors="pt")
        with torch.no_grad():
            bart_model.eval()
            pred = bart_model(input_ids=out["input_ids"], attention_mask=out["attention_mask"])
            embeddings = pred.last_hidden_state.mean(dim=1).reshape((-1))
            embeddings_list.append(embeddings)
    return torch.stack(embeddings_list)

df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

df_train = df.sample(frac=0.8, random_state=0)
print("df_train.shape: ", df_train.shape)
df_val = df.drop(index=df_train.index)
print("df_val.shape: ", df_val.shape)

# So the Countvectorizer previously turned our messages into numerical representations. Now we don't need it anymore. We can now use our convert_to_embeddings function here.
# df_train["message"] is a pandas series. We can just convert it to a python list with the tolist() function. Because we have only written our function to work with a normal Python list.
# print(df_train["message"])
# print(df_train["message"].tolist())
X_train = convert_to_embeddings(df_train["message"].tolist())
X_val = convert_to_embeddings(df_val["message"].tolist())

y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32).reshape((-1, 1))
y_val = torch.tensor(df_val["spam"].values, dtype=torch.float32).reshape((-1, 1))

print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_val.shape: ", X_val.shape)
print("y_val.shape: ", y_val.shape)

model = nn.Linear(768, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for i in range(0, 10000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0: 
        print(loss)

def evaluate_model(X, y): 
    model.eval()
    with torch.no_grad():
        y_pred = nn.functional.sigmoid(model(X)) > 0.25
        print("accuracy:", (y_pred == y).type(torch.float32).mean())
        
        print("sensitivity:", (y_pred[y == 1] == y[y == 1]).type(torch.float32).mean())
        
        print("specificity:", (y_pred[y == 0] == y[y == 0]).type(torch.float32).mean())

        print("precision:", (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean())
        # And we can see here how accurate and how high the sensitivity and the specificity is and also the precision. So the metrics here, they look outstanding um, compared to what we had before. So um, our idea of using the embeddings of an LM to grasp the meaning of a message actually works. And our metrics here are extremely good.

print("Evaluating on the training data")
evaluate_model(X_train, y_train)

print("Evaluating on the validation data")
evaluate_model(X_val, y_val)

# The reason is that now we are not working with the individual words. The individual words don't matter anymore. What now matters is the meaning that got captured through these embeddings.
# we can also see how our accuracies or sensitivities or specificity or the precision, how all of these metrics are significantly higher if we are working here with an LLM and the embeddings.
# So now even if here for example, we are using words that never occurred in our training data because we are not caring about the words. We are just caring about the meaning. We are now able to learn all of this, and this now really shows the power of the approach that we have here.
# now we could take data from, I don't know, 15 years ago or something like this, train a model on that, on the meaning of that and apply it to modern data.
X_custom = convert_to_embeddings([
    "We have release a new product, do you want to buy it?", 
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you come to the party?"
])

model.eval()
with torch.no_grad():
    pred = nn.functional.sigmoid(model(X_custom))
    print(pred)

# import sys
# import torch
# from torch import nn
# import pandas as pd
# from transformers import BartTokenizer, BartModel
# from tqdm import tqdm
# from sklearn.feature_extraction.text import CountVectorizer

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# bart_model = BartModel.from_pretrained("facebook/bart-base")

# def convert_to_embeddings(messages):
#     embeddings_list = []
#     for message in tqdm(messages):
#         out = tokenizer([message], 
#                         padding=True,
#                         max_length=512, 
#                         truncation=True, 
#                         return_tensors="pt")

#         with torch.no_grad():
#             bart_model.eval()

#             pred = bart_model(
#                 input_ids=out["input_ids"], 
#                 attention_mask=out["attention_mask"]
#             )
#             embeddings = pred.last_hidden_state.mean(dim=1)\
#                 .reshape((-1))
#             embeddings_list.append(embeddings)
#     return torch.stack(embeddings_list)

# df = pd.read_csv("./data/SMSSpamCollection", 
#                  sep="\t", 
#                  names=["type", "message"])

# df["spam"] = df["type"] == "spam"
# df.drop("type", axis=1, inplace=True)

# df_train = df.sample(frac=0.8, random_state=0)
# df_val = df.drop(index=df_train.index)

# X_train = convert_to_embeddings(df_train["message"].tolist())
# X_val = convert_to_embeddings(df_val["message"].tolist())

# y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32)\
#         .reshape((-1, 1))

# y_val = torch.tensor(df_val["spam"].values, dtype=torch.float32)\
#         .reshape((-1, 1))

# model = nn.Linear(768, 1)
# loss_fn = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# for i in range(0, 10000):
#     # Training pass
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = loss_fn(outputs, y_train)
#     loss.backward()
#     optimizer.step()

#     if i % 1000 == 0: 
#         print(loss)

# def evaluate_model(X, y): 
#     model.eval()
#     with torch.no_grad():
#         y_pred = nn.functional.sigmoid(model(X)) > 0.25
#         print("accuracy:", (y_pred == y)\
#             .type(torch.float32).mean())
        
#         print("sensitivity:", (y_pred[y == 1] == y[y == 1])\
#             .type(torch.float32).mean())
        
#         print("specificity:", (y_pred[y == 0] == y[y == 0])\
#             .type(torch.float32).mean())

#         print("precision:", (y_pred[y_pred == 1] == y[y_pred == 1])\
#             .type(torch.float32).mean()) 
        
# print("Evaluating on the training data")
# evaluate_model(X_train, y_train)

# print("Evaluating on the validation data")
# evaluate_model(X_val, y_val)

# X_custom = convert_to_embeddings([
#     "We have release a new product, do you want to buy it?", 
#     "Winner! Great deal, call us to get this product for free",
#     "Tomorrow is my birthday, do you come to the party?"
# ])

# model.eval()
# with torch.no_grad():
#     pred = nn.functional.sigmoid(model(X_custom))
#     print(pred)
