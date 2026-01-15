# import sys
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import transforms
# from torch.utils.data import random_split, DataLoader

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataset = torchvision.datasets.ImageFolder(
#     root='./images',
#     transform=preprocess
# )
# train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

dataset = torchvision.datasets.ImageFolder(
    root='./images',
    transform=preprocess
)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# resnet50_model = torchvision.models.resnet50(
#     weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
# )
# resnet50_model.fc = nn.Identity()
# resnet50_model.eval()

# for X, y in train_dataloader:
#     print(resnet50_model(X).shape)
#     print(resnet50_model(X))
#     break


# What we want to do is that we want to keep this network and also all the weights that we loaded. But what we want to do is that we want to discard a few layers here at the end. And all that we need to do is that we need to drop the last layer here, this fully connected layer.
resnet_50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
print(resnet_50_model)
for x in resnet_50_model.children():
    print("x: ", x)

print(resnet_50_model.fc) # we are able to access ".fc" because when we printed the model, we saw that the last layer is called "fc"

resnet_50_model.fc = nn.Identity() # I can just overwrite this layer. nn.Identity(): this will just create kind of like an empty layer that does nothing.

print(resnet_50_model)

for X,y in train_dataloader:
    outputs = resnet_50_model(X)
    print(outputs.shape) # torch.Size([32, 2048]) now we have 2048 features instead of 1000 features.
    break