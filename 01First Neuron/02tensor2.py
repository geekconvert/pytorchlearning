import torch 

b = torch.tensor(32)
w1 = torch.tensor(1.8)

X1 = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X1 * w1

# So this shape property allows you to identify the structure of the data that is in a tensor.
print(b.shape) # [] this means that this tensor here only holds a single value.
print(X1.shape) # [4] there are now here four elements in this tensor.
print(b.size())
print(X1.size())
print(y_pred[0]) # to access the first element of the tensor, But this is still in the tensor.
print(y_pred[1].item()) # to extract the value from the tensor.

# So this shape property allows you to identify the structure of the data that is in a tensor. size() and shape do tha same thing, they are equivalent.