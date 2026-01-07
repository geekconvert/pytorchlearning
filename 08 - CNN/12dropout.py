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

mnist_train = datasets.FashionMNIST(root='./data', download = True, train = True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download = True, train = False, transform=ToTensor())

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3,3), padding=1, padding_mode="reflect"),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Dropout(0.1) # we can use a dropout layer here, but we need to be careful to not drop out too much. if we are using a dropout within a convolutional block, it's usually best to keep this between 10 to 30% here.  the place of dropout matters. It's usually best practice to place dropout just after breaking the linearity meaning here after our rectified linear unit, and it's best practice to place a dropout out just after that.
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=(3,3), padding=1, padding_mode="reflect"),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Dropout(0.1)
    ),
    # And it turns out that the earlier we are in the network in general, the more general the features are that the network is able to focus on. When the network is still learning more general features. So I want to be a little bit more careful with the dropout. However, here between the last 100 neurons and the last ten output neurons, the features that it's learning are most high level. We want to make sure that we are not trusting a single advisor to put it back to the business context, or that we are not overly relying on an individual input and this is why dropout can go up to usually 50%. But the earlier we are in the network, the lower it should be. And especially for CNN layers, it should be even lower at maybe just 10 to 30%.
    nn.Flatten(),
    nn.Sequential(
        nn.Linear(64*7*7, 1000),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(100, 10)
    )
).to(device)

print(model)



loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# One thing that we do now need to be aware of is that now during training, the loss will be a little bit higher. The reason is that now here during training, our dropout will be applied. And this means that here when we are collecting an output from the model, and then we are applying our loss function to it. This loss will be higher because the model. Well, we had these connections that are just dropped. So the accuracy of the model during training won't be as good. However, once we are setting our model into evaluation mode, dropout is disabled. And in that case we should then see how well the model then performs. Again when we are creating the model. By default dropout will be enabled. It will be in training mode.

for i in range(0, 10):
    model.train() # this enables dropout
    
    loss_sum = 0
    for X,y in train_dataloader:
        y = F.one_hot(y, num_classes=10).type(torch.float32).to(device)
        X= X.to(device)

        optimizer.zero_grad()
        optputs = model(X)
        loss = loss_fn(optputs, y)
        loss.backward()
        optimizer.step()

        loss_sum+=loss.item()
    
    print(loss_sum)

model.eval() # this disables dropout
with torch.no_grad():
    accurate=0
    total =0
    for X,y in test_dataloader:
        X=X.to(device)
        y=y.to(device)
        outputs = nn.functional.softmax(model(X), dim=1)
        correct_pred = (y == outputs.max(dim=1).indices)
        total+=correct_pred.size(0)
        accurate+=correct_pred.type(torch.int).sum().item()

    print("Accuracy on validation data:", accurate / total)