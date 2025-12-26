import torch
from torch import nn
import pandas as pd


df = pd.read_csv("./data/student_exam_data.csv")

X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values, 
    dtype=torch.float32
)

y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1, 1))
print(X.shape, y.shape)

model = nn.Linear(2, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for i in range(0, 500000):
    # before each training step so that, um, the gradient calculations that happen while we are applying or while we are running the model, that they are being reset after every turn. So this we need to tell our optimizer.
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    #  the loss being back propagated.
    loss.backward()
    # And then we want to ask the optimizer to do a step into the right direction to optimize the weights.
    optimizer.step()

    if i % 10000 == 0:
        print(loss)

model.eval()

# torch, no need to keep track of the gradients anymore. We are not training anymore.
with torch.no_grad():
    y_pred = nn.functional.sigmoid(model(X)) > 0.5
    # as data type of y is float32, we need to convert y_pred to float32 too for comparison
    y_pred_correct = y_pred.type(torch.float32) == y
    # convert the type here again into the true false values into a float32.
    print(y_pred_correct.type(torch.float32).mean())

# import torch
# from torch import nn
# import pandas as pd

# df = pd.read_csv("./data/student_exam_data.csv")

# X = torch.tensor(
#     df[["Study Hours", "Previous Exam Score"]].values, 
#     dtype=torch.float32
# )
# y = torch.tensor(df["Pass/Fail"], dtype=torch.float32)\
#     .reshape((-1, 1))

# model = nn.Linear(2, 1)
# loss_fn = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# for i in range(0, 500000):
#     optimizer.zero_grad()
#     outputs = model(X)
#     loss = loss_fn(outputs, y)
#     loss.backward()
#     optimizer.step()

#     if i % 10000 == 0:
#         print(loss)

# model.eval()
# with torch.no_grad():
#     y_pred = nn.functional.sigmoid(model(X)) > 0.5
#     y_pred_correct = y_pred.type(torch.float32) == y
#     print(y_pred_correct.type(torch.float32).mean())
