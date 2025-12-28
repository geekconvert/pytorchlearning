import torch 

# This will be the shape of the data that our model will require from us.
# we only have one individual input per entry.
X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
])
print(X)
print(X.shape)
print(X.size())
print(X.size(0)) # The advantage here of size is that we can also pass in the dimension of interest for us.
print(X.size(1))
print(X[0, 0])
print(X[1, :])# slices. all elements in the 2nd row
print(X[:, 0])


# Be aware um deviating from that in a single row would not be allowed.
# X = torch.tensor([
#     [10, 5], 
#     [38], 
#     [100], 
#     [150]
# ])