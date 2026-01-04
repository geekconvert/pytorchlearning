import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from PIL import Image

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

mnist_train = datasets.MNIST(root='./data', download=True, train=True, transform=ToTensor())

# We can see that the background needs to be zeros and the foreground needs to be filled with ones.
# print(mnist_train[0])

model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 10))
model.load_state_dict(torch.load('mnist-model.pth', weights_only=True))

img = Image.open("3.png")
img.thumbnail((28, 28))
img = img.convert('L')

t = ToTensor()

# After this calculation then the background will now be consisting out of zeros. And the foreground is now going to consist out of one.
X = (1 - t(img).reshape((-1, 784)))  # Invert colors: MNIST has white digits on black background
# print(X)
# print(X.shape)
outputs = model(X)

# The data that we want to utilize the model on needs to match the format that we had trained the model with. This correctly classifies the digit 3 now.
print(nn.functional.softmax(outputs, dim=1))

# import sys
# import torch
# from torch.utils.data import DataLoader
# from torch import nn
# import torch.nn.functional as F

# from PIL import Image

# import torchvision.datasets as datasets
# from torchvision.transforms import ToTensor

# # mnist_train = datasets.MNIST(root='./data', download=True, train=True, transform=ToTensor())
# # print(mnist_train[0])

# model = nn.Sequential(
#     nn.Linear(784, 50),
#     nn.ReLU(),
#     nn.Linear(50, 50),
#     nn.ReLU(),
#     nn.Linear(50, 10)
# )
# model.load_state_dict(torch.load('mnist-model.pth', weights_only=True))

# img = Image.open("3.png")
# img.thumbnail((28, 28))
# img = img.convert('L')

# t = ToTensor()
# X = (1 - t(img).reshape((-1, 784)))
# print(X)
# print(X.shape)
# outputs = model(X)
# print(nn.functional.softmax(outputs, dim=1))
