import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Running on device:", device)

mnist_train = datasets.FashionMNIST(root='./data', download = True, train = True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download = True, train = False, transform=ToTensor())

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=(3,3), padding=1, padding_mode="reflect"),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(588, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(0, 10):
    model.train()

    loss_sum=0
    for X, y in train_dataloader:
        # I only send 32 images at a time, and the corresponding y values to it to my respective device.
        y = F.one_hot(y, num_classes=10).type(torch.float32).to(device)
        X = X.to(device)

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
    total =0
    for X, y in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        outputs = nn.functional.softmax(model(X), dim=1)
        correct_pred = (y == outputs.max(dim=1).indices)
        total+=correct_pred.size(0)
        accurate+=correct_pred.type(torch.int).sum().item()

    print("Accuracy on validation data:", accurate / total)

# Despite using MPS it took more time. but then also you can see that it used less CPU resources overall. Copying the data to the device here actually takes up a lot of time and a lot of processing time. 

# If we were to unplug MacBook and run this on battery power, my battery would last significantly longer, my laptop would stay quieter and all of these things. So there are also these additional benefits of utilizing hardware acceleration.

# However, again, this code here would scale significantly better because we are using less computational resources.