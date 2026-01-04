import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor

from PIL import Image

model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 10))
model.load_state_dict(torch.load('mnist-model.pth', weights_only=True))

# Important here the size must be exactly 28 times 28 pixels because otherwise, um, well, the model has not been trained on any other data, so, um, we are not going to be able to then apply the model also because, um, we have 784 inputs, meaning 28 times 28. So this must be exactly this size.
img = Image.open("4.png")

# If it's not this size, you can also just say here img or image dot thumbnail. Um, and then you could be like okay, here you want to resize it to 28 times 28 pixels. And this will then change this image here. Um, and it will then be resized.
img.thumbnail((28, 28))

# By the way in case it had multiple color channels, maybe red, green and blue. In that case image shape would be 3,28,28. And in that case you would need to say something like this image dot convert. And then you would need to say L. And this one here would then stand for grayscale this convert method. We can also see it here does return a new image. So this is creating a copy of the image where everything is converted into grayscale.
img = img.convert('L')

t = ToTensor()
X = t(img).reshape((-1, 784))
print(X.shape)
outputs = model(X)

# Currently the output is saying it is a 5 but it is 4 actually
print(nn.functional.softmax(outputs, dim=1))



# import sys
# import torch
# from torch.utils.data import DataLoader
# from torch import nn
# import torch.nn.functional as F

# from torchvision.transforms import ToTensor

# from PIL import Image

# model = nn.Sequential(
#     nn.Linear(784, 50),
#     nn.ReLU(),
#     nn.Linear(50, 50),
#     nn.ReLU(),
#     nn.Linear(50, 10)
# )
# model.load_state_dict(torch.load('mnist-model.pth', weights_only=True))

# img = Image.open("4.png")
# img.thumbnail((28, 28))
# img = img.convert('L')

# t = ToTensor()
# X = t(img).reshape((-1, 784))
# print(X.shape)
# outputs = model(X)
# print(nn.functional.softmax(outputs, dim=1))
