import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer



df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv= CountVectorizer(max_features=1000)
messages = cv.fit_transform(df['message'])

X = torch.tensor(messages.todense(), dtype=torch.float32)
y = torch.tensor(df["spam"], dtype=torch.float32).reshape((-1, 1))

print(X)
print(X.shape)
print(y)
print(y.shape)

model = nn.Linear(1000, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for i in range(0, 10000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0: 
        print(loss)


# If the model or if the predictions that our model makes should be considered probabilities, um, we can't use a value like this -1.4%. This is not a valid probability. All the probabilities must be between 0 and 1. tensor(-0.6452) and tensor(1.5441) are max and min values that our model is predicting here.
model.eval()
with torch.no_grad():
    y_pred = model(X)
    print(y_pred)
    print(y_pred.min())
    print(y_pred.max())



# import torch
# from torch import nn
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer

# df = pd.read_csv("./data/SMSSpamCollection", 
#                  sep="\t", 
#                  names=["type", "message"])

# df["spam"] = df["type"] == "spam"
# df.drop("type", axis=1, inplace=True)

# cv = CountVectorizer(max_features=1000)
# messages = cv.fit_transform(df["message"])

# X = torch.tensor(messages.todense(), dtype=torch.float32)
# y = torch.tensor(df["spam"], dtype=torch.float32)\
#         .reshape((-1, 1))

# model = nn.Linear(1000, 1)
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# for i in range(0, 10000):
#     # Training pass
#     optimizer.zero_grad()
#     outputs = model(X)
#     loss = loss_fn(outputs, y)
#     loss.backward()
#     optimizer.step()

#     if i % 100 == 0: 
#         print(loss)

# model.eval()
# with torch.no_grad():
#     y_pred = model(X)
#     print(y_pred)
#     print(y_pred.min())
#     print(y_pred.max())