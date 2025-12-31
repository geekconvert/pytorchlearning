import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

X =torch.tensor(messages.todense(), dtype=torch.float32)
y = torch.tensor(df["spam"], dtype=torch.float32).reshape(-1, 1)

model = nn.Linear(1000, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for i in range(25000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(loss)

model.eval()
with torch.no_grad():
    #  let's say the specificity is not that important. Meaning it's okay if every now and then a non-spam message, uh, gets detected as spam, then it could be, for example, here I could change the percentage to 0.25 and everything where the model is more confident than 25% that it is spam, we will now consider as spam.
    # y_pred = nn.functional.sigmoid(model(X)) > 0.5
    y_pred = nn.functional.sigmoid(model(X)) > 0.25
    # print(y_pred)
    # print(y)
    # y = (y == 1) no need to convert y to boolean as without it also same result.

    """
    y_pred[y == 1] uses boolean indexing: it keeps only the entries in y_pred where the corresponding entry in y equals 1.

    Example:

    y_pred = tensor([0.2, 0.8, 0.6, 0.1])
    y = tensor([0., 1., 1., 0.])
    y == 1 → tensor([False, True, True, False])

    y_pred[y == 1] → tensor([0.8, 0.6]) (only predictions for the positive class).
    """

    # We now have some measurements here that would allow us to actually tune th cutoff value 0.25 based on what we want to optimize for.
    print("accuracy:", (y_pred == y).type(torch.float32).mean())
    print("sensitivity:", (y_pred[y == 1] == y[y == 1]).type(torch.float32).mean())
    print("specificity:", (y_pred[y == 0] == y[y == 0]).type(torch.float32).mean())
    print("precision:", (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean()) 

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
# loss_fn = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# for i in range(0, 10000):
#     # Training pass
#     optimizer.zero_grad()
#     outputs = model(X)
#     loss = loss_fn(outputs, y)
#     loss.backward()
#     optimizer.step()

#     if i % 1000 == 0: 
#         print(loss)

# model.eval()
# with torch.no_grad():
#     y_pred = nn.functional.sigmoid(model(X)) > 0.25
#     print("accuracy:", (y_pred == y)\
#         .type(torch.float32).mean())
    
#     print("sensitivity:", (y_pred[y == 1] == y[y == 1])\
#         .type(torch.float32).mean())
    
#     print("specificity:", (y_pred[y == 0] == y[y == 0])\
#         .type(torch.float32).mean())

#     print("precision:", (y_pred[y_pred == 1] == y[y_pred == 1])\
#         .type(torch.float32).mean()) 
   