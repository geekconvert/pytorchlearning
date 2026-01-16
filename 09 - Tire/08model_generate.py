# import sys
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import transforms
# from tqdm import tqdm
# from torch.utils.data import random_split, DataLoader

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.mps.is_available():
#     device = torch.device("mps")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Using device: ", device)

# preprocess = transforms.Compose([
#     transforms.Resize(512),
#     transforms.RandomRotation(10),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# it now took significantly longer to train. So I guess all of these additional operations here, they do take additional processing power.
# So all of these transformations take a lot of time, especially if the images are large.
# preprocess = transforms.Compose([
#     transforms.RandomRotation(10),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# so we can change our pre-processing pipeline to first resize our images, make them a little bit smaller, just so that these transformations here don't take that much time anymore.
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomRotation(10),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# dataset = torchvision.datasets.ImageFolder(
#     root='./images',
#     transform=preprocess
# )

# train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)



# when we generating here my data set, it's not like the pre-processing pipeline is being applied to the data. Later on when we will ask for data then "transform=preprocess" this is how you will pre-process the data.
dataset = torchvision.datasets.ImageFolder(
    root='./images',
    transform=preprocess
)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# every time that we are going to iterate over our train_dataloader, the pre-processing pipeline will only be applied at this point in time. And this means that every time we are going over our training data, for example, a different permutation or a different image will be generated here by our pre-processing pipeline.

# resnet50_model = torchvision.models.resnet50(
#     weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
# )
# resnet50_model.fc = nn.Identity()
# for param in resnet50_model.parameters():
#     param.requires_grad = False
# resnet50_model.eval()
# resnet50_model = resnet50_model.to(device)

resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = nn.Identity()
for param in resnet50_model.parameters():
    param.requires_grad = False

resnet50_model.eval()
resnet50_model = resnet50_model.to(device)

# fc_model = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 1)
# )
# fc_model = fc_model.to(device)
# model = nn.Sequential(
#     resnet50_model,
#     fc_model
# )
# model = model.to(device)

fc_model = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1)
)
fc_model = fc_model.to(device)
model = nn.Sequential(
    resnet50_model,
    fc_model
)
model = model.to(device)

# optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.00025)
# loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.00025)
loss_fn = nn.BCEWithLogitsLoss()

# This operation is not allowed because there is not really a first batch. It's being generated on the fly when we are iterating over it with a for loop. And this is why this notation here is not supported. And we can only iterate over our train data loader with a for loop, because then the whole pre-processing pipeline is being invoked on the fly.
# print(train_dataloader[0]) #  'DataLoader' object is not subscriptable

# for epoch in range(15):
#     print(f"--- EPOCH: {epoch} ---")
#     model.train()
#     resnet50_model.eval()
    
#     loss_sum = 0
#     train_accurate = 0
#     train_sum = 0
#     for X, y in tqdm(train_dataloader):
#         X = X.to(device)
#         y = y.to(device).type(torch.float).reshape(-1, 1)

#         outputs = model(X)
#         optimizer.zero_grad()
#         loss = loss_fn(outputs, y)
#         loss_sum+=loss.item()
#         loss.backward()
#         optimizer.step()

#         predictions = torch.sigmoid(outputs) > 0.5
#         accurate = (predictions == y).sum().item()
#         train_accurate+=accurate
#         train_sum+=y.size(0)
#     print("Training loss: ", loss_sum / len(train_dataloader))
#     print("Training accuracy: ", train_accurate / train_sum)


#     torch.save(fc_model.state_dict(), f"fc_model_{epoch}.pth")

#     model.eval()
#     val_loss_sum = 0
#     val_accurate = 0
#     val_sum = 0
#     with torch.no_grad():
#         for X, y in tqdm(val_dataloader):
#             X = X.to(device)
#             y = y.to(device).type(torch.float).reshape(-1, 1)

#             outputs = model(X)
#             loss = loss_fn(outputs, y)
#             val_loss_sum+=loss.item()

#             predictions = torch.sigmoid(outputs) > 0.5
#             accurate = (predictions == y).sum().item()
#             val_accurate+=accurate
#             val_sum+=y.size(0)
#     print("Validation loss: ", val_loss_sum / len(val_dataloader))
#     print("Validation accuracy: ", val_accurate / val_sum)


# execution or training may take even a little bit longer because now here of course all of these image operations here will have to be performed. And this will take additional processing pipeline steps or CPU cycles.
for epoch in range(15):
    print(f"--- EPOCH: {epoch} ---")
    model.train()
    resnet50_model.eval()

    loss_sum = 0
    train_accurate = 0
    train_sum = 0

    for X, y in tqdm(train_dataloader):
        X = X.to(device)
        y = y.to(device).type(torch.float).reshape(-1, 1)

        outputs = model(X)
        optimizer.zero_grad()
        loss = loss_fn(outputs, y)
        loss_sum+=loss.item()
        loss.backward=()
        optimizer.step()

        predictions = torch.sigmoid(outputs) > 0.5
        accurate = (predictions == y).sum().item()
        train_accurate+=accurate
        train_sum+=y.size(0)

    print("Training loss: ", loss_sum / len(train_dataloader))
    print("Training accuracy: ", train_accurate / train_sum)

    torch.save(fc_model.state_dict(), f"fc_model_{epoch}.pth")


    #  We can see here now also that the validation loss is fluctuating a bit more. The reason is that also for the validation data, the training or sorry the pre-processing pipeline is being applied. So also there we have random rotations, random vertical flips and so on. Um, they are, of course, just adding additional noise there.
    model.eval()
    val_loss_sum = 0
    val_accurate = 0
    val_sum = 0
    with torch.no_grad():
        for X,y in tqdm(val_dataloader):
            X = X.to(device)
            y = y.to(device).type(torch.float).reshape(-1, 1)

            outputs = model(X)
            loss = loss_fn(outputs, y)
            val_loss_sum+=loss.item()

            predictions = torch.sigmoid(outputs) > 0.5
            accurate = (predictions == y).sum().item()
            val_accurate+=accurate
            val_sum+=y.size(0)

    print("Validation loss: ", val_loss_sum / len(val_dataloader))
    print("Validation accuracy: ", val_accurate / val_sum)