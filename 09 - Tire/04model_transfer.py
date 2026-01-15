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
    transform = preprocess
)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)



# resnet50_model = torchvision.models.resnet50(
#     weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
# )
# resnet50_model.fc = nn.Identity()
# for param in resnet50_model.parameters():
#     param.requires_grad = False
# resnet50_model.eval()

resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = nn.Identity()
for param in resnet50_model.parameters():
    param.requires_grad = False # this is just a property to all the parameters here that we are setting. And this now just tells PyTorch that this parameter does not require gradient tracking. Because even in evaluation mode, PyTorch would keep track of the gradients of the slope with respect to each parameter here. And we don't want this because then, um, applying this model to data will just be faster.
resnet50_model.eval() # ResNet model should be put into evaluation mode because we don't want any training features there to be applied.

# fc_model = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 1)
# )

# model = nn.Sequential(
#     resnet50_model,
#     fc_model
# )


# And the big decision is now is whether we want this ResNet model here at the beginning or whether we want to train it or not. And if we would train it, then we would need to adjust the weights there. But you can imagine this is now ResNet 50. So it's 50 layers. So it's quite deep. So training there may take quite a while. Um, also back propagation through all of these 50 layers will take quite some time. Um, we can argue it could lead to better results if we are willing to fine tune the ResNet model as well.

#  the approach that We will be choosing throughout this chapter here, which will mean that we just keep the ResNet model in place. We are just using it for feature extraction, meaning that we take our images and turn them into 2048 activations. But we are not going to train this model at all. We are going to lock the parameters of this model here in place. This will make training significantly faster because only here our fully connected model needs to be trained, whereas our ResNet model will just stay fixed and it will just be applied.

# Why am I attaching two layers here and not just one? Why not just go with one linear layer directly from 2048 inputs into one output. Well, my feeling is that this would be a little bit much because later on we will be only training this model here. We will not be training the ResNet model. So we still may need some additional flexibility. And also quite often it's quite common that we don't turn 2048 neurons directly into a single neuron, but that we have like 1 or 2 steps in between to give the model a bit more flexibility to fit the data.

fc_model = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1)
)

model = nn.Sequential(
    resnet50_model,
    fc_model
)

# optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.001)
# print(model)

# for X, y in train_dataloader:
#     out = model(X)
#     print(out.shape)
#     break

optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.001)
print(model)

for X, y in train_dataloader:
    # When we are applying this model here, by default, PyTorch keeps track of the gradients during the last execution. And this was the reason for example, when we are just trying to make a prediction, we would need to say with torch.no_grad. However, we do need to keep track of the gradients because one part of the model we want to train and this part of the model we don't want to train. So we can't in general say torch.no_grad(). We need to disable the gradient tracking only here for the ResNet50 model, so that calculations are being performed more efficiently.
    out = model(X)
    print(out.shape)
    break



