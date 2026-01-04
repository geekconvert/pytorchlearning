# import sys
# import torch
# from torch.utils.data import DataLoader
# from torch import nn
# import matplotlib.pyplot as plt

# import torchvision.datasets as datasets
# from torchvision.transforms import ToTensor

# mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
# mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

# train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

# model = nn.Sequential(
#     nn.Conv2d(1, 3, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(2352, 100),
#     nn.ReLU(),
#     nn.Linear(100, 10)
# )

# image = mnist_train[0][0].reshape(1, 1, 28, 28)
# output = model(image)
# print(output.shape)

import torch 
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

# model = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3,3), padding=1, padding_mode="reflect"))
# print(mnist_train[0][0].shape)

# This new dimension here is this is the first image that we want to predict or only one image is in there that we want to predict. But we could have here in this dimension multiple images. Second dimension here would be the color channel here in this case grayscale. And then we have the images in the size of 28 times 28 pixels. But this one for example would now allow me to or this structure here to have multiple images in here.
print(mnist_train[0][0].reshape(1, 1, 28, 28).shape)

# In this case our images are in grayscale, so it's  1, and the out channels are 3. This kernel size we can set to (3,3). And this just defines how large our filter is.
# applying filter in Conv2d layer reduce the size of image so we use padding to keep the size same as input image. Padding mode reflect means that the values at the border are reflected(copied) to create new values for padding.
model = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=(3,3), padding=1, padding_mode="reflect"),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2352, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

image = mnist_train[0][0].reshape(1, 1, 28, 28)
output = model(image)
print(output.shape)