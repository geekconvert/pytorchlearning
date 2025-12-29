# b = 32
# w1 = 1.8

# x1 = 100

# y_pred = 1 * b + x1 * w1
# print(y_pred)


###############

# import torch 

# b = torch.tensor(32)
# w1 = torch.tensor(1.8)

# x1 = torch.tensor(100)

# y_pred = 1 * b + x1 * w1
# print(y_pred)


###############

# import torch 

# b = torch.tensor(32)
# w1 = 1.8

# x1 = torch.tensor(100)

# # And this w1 is not a tensor in this case um the tensor will intercept this multiplication and it will still create a new tensor here. this multiplication on one side, there's a tensor, then the result will also be a tensor. And the same applies to additions. At least on one of the two sides of the plus, there's a tensor. Then the whole result will also be a tensor.

# y_pred = 1 * b + x1 * w1
# print(y_pred)


###############

# import torch 

# b = 32
# w1 = 1.8

# x1 = torch.tensor(100)

# y_pred = 1 * b + x1 * w1
# print(y_pred)


###############


import torch 

# tensors that hold individual values. These are also called scalars.
b = torch.tensor(32)
print("b: ", b)
w1 = torch.tensor(1.8)
print("w1: ", w1)

# And it turns out that even now, even if now we are not utilizing a GPU. Because even if we are running this now on the CPU, this all these calculations are more efficient because the CPU has some built in mechanisms that allow it to to apply the same calculation to multiple data points at the same time. Some so-called vector extensions.
# for PyTorch, this is now already, um, it's something that is now already being optimized and it's leveraging now here advanced features from the CPU that normal Python code could not really utilize, because there we would perform one calculation after another. And here we are performing all these multiplications at once.

#  tensors that hold a vector, meaning that we just hold a list of multiple values.
X1 = torch.tensor([10, 38, 100, 150])
print("X1: ", X1)
print("X1 shape: ", X1.shape)

y_pred = 1 * b + X1 * w1
print("y_pred: ", y_pred)


# Tensors are a core concept in PyTorch. But they are pretty much just a box in which values are being stored. This enables efficient data handling because this makes our calculations a bit more efficient because this box can be stored, for example, on optimized hardware such as GPUs, and calculations can then be performed there. And the big advantage is that us as programmers, we can just write normal Python code, and PyTorch just happens to store the actual data on the optimized hardware. But we don't, for example, need to write the code that actually runs on the GPU. And this makes it relatively easy to write GPU optimized code. Because of this, uh, these tensors significantly speed up deep learning training, meaning that we want to extract meaning from data, um, and inference, which means that we want to apply a model to new data.

#there are different versions of a tensor. So for example, a tensor could just hold a single value. Or you can see here multiple values or even a matrix.