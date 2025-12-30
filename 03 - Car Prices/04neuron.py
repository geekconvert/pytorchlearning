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
# print("X: ", X)
y = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
# print("y: ", y)

model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0000000001)

# We can see here that we are getting this Nan here. But we can also see here how these parameters here kind of explode. So probably these parameters here are just getting way too large and we are not really able to handle them anymore. And then either parameters, um, become too large that they can't be stored anymore. Or what can also happen is that we end up dividing by zero, because when we take the derivative or something and then this Nan happens.
#The reason can be that our learning rate is too high. So we can try to reduce our learning rate here.
for i in range(0, 1000):
    #training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0: 
        print(model.bias)
        print(model.weight)
        print(loss)

prediction = model(X)
print(prediction)

prediction = model(torch.tensor([[5.0, 10000.0]]))
print(prediction)

# currently output of this is greater than above one, which is not correct as price should decrease with increase in milage
prediction = model(torch.tensor([[5.0, 20000.0]]))
print(prediction)
