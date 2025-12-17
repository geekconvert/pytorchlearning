import torch 
from torch import nn

# often pytorch takes care of data type conversion but here we are running the model. This is a more complex calculation. And also when we are trying to trying to learn behaviors or parameters, we would need to run the model over and over and over again. Maybe, for example, 10,000 times. And if this were to be converted automatically, we would convert the data from this integer structure to a floating point tensor. Let's say 10,000 times. And this would be highly inefficient.
# RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float
X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
])

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
], dtype=torch.float32)

model = nn.Linear(1, 1)

# for the learning process, it's required that the tensor here for our bias and for our weight needs to be a floating point number.
# model.bias = nn.Parameter(
#     torch.tensor([32])
# )

model.bias = nn.Parameter(
    torch.tensor([32], dtype=torch.float32)
)

model.weight = nn.Parameter(
    torch.tensor([[1.8]], dtype=torch.float32)
)

print(model.bias)
print(model.weight)

y_pred = model(X)
print(y_pred)