import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Running on device:", device)


mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

# But because now here our structure is getting more and more complex, it's usually best practice to split this into multiple sequential models, and in sequential we can just put or wrap things into yet another sequential here.
model = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=(3,3), padding=1, padding_mode="reflect"), #1 channel input, 3 channel output
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU()
    ),
    nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=(3,3), padding=1, padding_mode="reflect"),# 3 channel input, 6 channel output
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
    ),
    nn.Flatten(),
    nn.Sequential(
        nn.Linear(6*7*7, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
).to(device)
print(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(0, 10):
    model.train()

    loss_sum = 0
    for X,y in train_dataloader:
        X = X.to(device)
        y = F.one_hot(y, num_classes=10).type(torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        loss_sum+=loss.item()
    
    print(loss_sum)


model.eval()
with torch.no_grad():
    accurate = 0
    total = 0
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        outputs = nn.functional.softmax(model(X), dim=1)
        correct_pred = (y == outputs.max(dim=1).indices)
        total+=correct_pred.size(0)
        accurate+=correct_pred.type(torch.int).sum().item()
    
    print("Accuracy on validation data:", accurate / total)