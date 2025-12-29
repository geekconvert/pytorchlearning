import torch
from torch import nn

# The idea with batch learning is that we want to evaluate both of these inputs at the same time, and compare it to the corresponding outputs.
# Input: Temperature in °C
X = torch.tensor([
    [10],
    [37.78]
], dtype=torch.float32)

# Actual value: Temperature °F
y = torch.tensor([
    [50],
    [100.0]
], dtype=torch.float32)

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# I can have my model applied to both entries at the same time. And I can then also have my loss applied to both entries at the same time.
# Learning should be a little bit faster. We should usually be able to, for example, utilize GPUs better because we are running more computations at the same time, which increases the efficiency. So this is usually a good approach to always bundle multiple entries together.
# it may happen that for the same number of digits after the decimal point we might need a few more iterations here. Because the overall number of steps that we do are still less than before.
for i in range(0, 150000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0: 
        print(model.bias)
        print(model.weight)

print("----")

# And you can see here 37.5°C is roughly about 100°F. And all of this has now been learned from our example data. And this now really shows, um, how amazing this technology now already is. Because this parameters(weight and bias) here, um, had to be learned from just the data. And the best thing is that this would now also scale, for example, to different types of data.
measurements = torch.tensor([
    [37.5]
], dtype=torch.float32)

# This just means that I want to turn off features that might be specific for training because there might be some features that behave differently during training and during inference.
model.eval()

# And this with block just tells PyTorch and the optimizers there, um, that we don't need to keep track of the gradients because that's not needed if we have finished training and we just want our model to be to be applied so we can disable this to increase performance.
with torch.no_grad():
    prediction = model(measurements)
    print(prediction)


