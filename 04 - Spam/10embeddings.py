import torch
from torch import nn
from transformers import BartTokenizer, BartModel

# The first step is that I need to run these through a tokenizer. And then I can run them through an LLM to get the actual embeddings.
messages = [
    "We have release a new product, do you want to buy it?", 
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you come to the party?",
    "hello"
]

# So "facebook/bart-base" here is a tokenizer, um, or a model with a associated tokenizer with it that Facebook released for us.
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# But here the number of tokens is different now between each message. Each each message can have a different number of tokens. And it turns out that this is a little bit more tricky for training or to use for the model.
# output = tokenizer(messages)
# print(output)

# "padding=True" means that if, for example, we have a shorter message here that then this message will just get padded. The padding tokens will be added to the end of the message so that all messages have the same length and those are just ignored by the model.
# And this is why we got this attention mask here that just says like these tokens, please ignore them. But then our inputs always have the same shape and the same number of tokens and then it's easier to train.
# The next parameter that I should set is the max length parameter here that we can set, for example, to 512. That means that each of the messages can consist out of a maximum of 512 tokens. And otherwise we can be like truncation. We can just set this one to true. We can just be like, okay, if it's longer, then we are just not going to bother with the rest. So this doesn't mean that it's 512 words but 512 tokens.
# return_tensors="pt" :  So now we are going to not just get like this is so far a numpy array. That's another data science library for storing numerical data. But here we want it as a PyTorch tensor.
out = tokenizer(messages, 
                    padding=True, 
                    max_length=512, 
                    truncation=True, 
                    return_tensors="pt")

print("out: ", out)

# So I can use a model that is provided by Facebook. And this already implements or this will now fetch the weights and everything that is needed for all the different layers, so that I can just use this model.
bart_model = BartModel.from_pretrained("facebook/bart-base")

with torch.no_grad():
    bart_model.eval()
    # Well, it turns out "out" here was a dictionary that, uh, let me just execute it again. Um, that consisted here "out" of two keys, input_ids and the attention_mask. And both of these are just being converted into parameters here. Either way is ok.
    # pred = bart_model(**out)
    # bart model returned a lot of information , but we are only interested in the last hidden state
    pred = bart_model(
        input_ids=out["input_ids"], 
        attention_mask=out["attention_mask"]
    )
    # We got 16 in the next dimension, and then and then 768 in the last dimension. Well, our LLM it only predicted one token at a time, and it had to run, 16 times, because here the longest number of tokens had been 16. So this is why we are getting the individual activations there for each of these 16 steps. 768 is the number of output values that we got there. so in each step we have 768, uh, numerical values that represent the meaning of that.
    # pred.last_hidden_state had shape torch.Size([4, , 16 , 768]) but with pred.last_hidden_state.mean(dim=1) it is torch.Size([4, 768])
    # Where did these 16 entries for each of the messages um, go to? Well, we took the average there, um, so that everything is now a simple two dimensional structure.
    embeddings = pred.last_hidden_state.mean(dim=1)
    print(embeddings.shape)

    # first row give all columns
    print(embeddings[0, :])

# from transformers import BartTokenizer, BartModel
# import torch
# from torch import nn

# messages = [
#     "We have release a new product, do you want to buy it?", 
#     "Winner! Great deal, call us to get this product for free",
#     "Tomorrow is my birthday, do you come to the party?",
# ]

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# out = tokenizer(messages, 
#                 padding=True, 
#                 max_length=512, 
#                 truncation=True, 
#                 return_tensors="pt")

# bart_model = BartModel.from_pretrained("facebook/bart-base")
# with torch.no_grad():
#     bart_model.eval()
#     print(out)

#     pred = bart_model(
#         input_ids=out["input_ids"], 
#         attention_mask=out["attention_mask"]
#     )
#     embeddings = pred.last_hidden_state.mean(dim=1)
#     print(embeddings.shape)
#     print(embeddings[0, :])