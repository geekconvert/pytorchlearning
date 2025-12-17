import torch 
from torch import nn

# .0 is important here, otherwise code won't work.
X = torch.tensor([
    [10.0], 
    [38.0], 
    [100.0], 
    [150.0]
])

# I want to create a linear neuron.
# one input and we want to create one linear neuron. So here we have one input because we only have one feature that we are looking at here one column column in our data. And we have one output here or one neuron that we want to create.
model = nn.Linear(1, 1)
print(model)

# model and weight are initialized randomly. This is how deep learning works. We initialize things randomly and then we try to optimize things.
print(model.bias)
print(model.weight)

y_pred = model(X)
print(y_pred)

# This just turns a tensor into a parameter that we can then work with. This needs to be in the same shape as the original bias and weight. Thats why [32.0] and [1.8]]
model.bias = nn.Parameter(
    torch.tensor([32.0])
)
model.weight = nn.Parameter(
    torch.tensor([[1.8]])
)

print(model.bias)
print(model.weight)

y_pred = model(X)
print(y_pred)