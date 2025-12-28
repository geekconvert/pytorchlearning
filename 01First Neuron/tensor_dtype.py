import torch 
from torch import nn

print(5)
print(5.5)
print(type(5))
print(type(5.5))

# a pytorch tensor is typed.
X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
])
print(X.dtype) #int64

# Quite often also PyTorch does a good job to abstract this datatype conversion away from us. The the data type here will be converted into a floating point data type automatically.
X = X * 0.5
print(X)
print(X.dtype) # float32

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
], dtype=torch.float32)
print(X)
print(X.dtype)

X = X.type(torch.int64)
print(X)
print(X.dtype)
# result = X * 0.5
# print(result)
# print(result.dtype)
