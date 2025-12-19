import pandas as pd
import torch
from torch import nn

# Pandas: Reading the data
df = pd.read_csv("./data/used_cars.csv")

# Pandas: Preparing the data
age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "")
milage = milage.str.replace(" mi.", "")
milage = milage.astype(int)

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

# print(age)
# print(milage)
# print(price)

# # this is not yet the right way how this needs to be, because the way our torch tensor would need to be structured would be that we would need to have a matrix, and then there would need to be the data in it.
# z = torch.tensor(age)
# y = torch.tensor(milage)
# print(z)
# print(y)

# # even this is not the right way because this is going to create a 2D tensor, but what we really want is a matrix where each row is one data point, and the first column is age, and the second column is milage.
# a = torch.tensor([age, milage])
# print(a)

X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32), 
    torch.tensor(milage, dtype=torch.float32)
])
model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)

prediction = model(X)
print(prediction)

# Torch: Creating X and y data (as tensors)
# X = torch.column_stack([
#     torch.tensor(age, dtype=torch.float32),
#     torch.tensor(milage, dtype=torch.float32)
# ])

# model = nn.Linear(2, 1)
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)

# prediction = model(X)
# print(prediction)