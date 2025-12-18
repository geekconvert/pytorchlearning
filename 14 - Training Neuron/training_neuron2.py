import torch
from torch import nn

# Input: Temperature in °C
X1 = torch.tensor([[10]], dtype=torch.float32) 
# Actual value: Temperature °F
y1 = torch.tensor([[50]], dtype=torch.float32) 

# Input: Temperature in °C
X2 = torch.tensor([[37.78]], dtype=torch.float32) 
# Actual value: Temperature °F
y2 = torch.tensor([[100.0]], dtype=torch.float32) 

# It's important here that we train our model or our neuron with both data points, because otherwise it's not going to be able to find the correct values for w1 or B, it's just mathematically not possible.

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# running it many times, then it will actually be able to approach the ideal parameters here.
# Sometimes Nan can come. Not a number(NaN) means that there's a numerical problem in our calculations. Um, maybe, for example, we once tried to divide by zero or something like this, and then everything from then on will be not a number or we exceeded the, the number range that can be stored. And then it's also not a number. So it just means that there's a numerical problem here. Usually this happens because the learning rate is too great and then the gradients might actually collapse to zero. And then we end up dividing by that can cause issue or get too large and um, then also we end up at not a number.So we will just reduce the learning rate.

# If you are wondering why does it take so long and it's it's just such a simple formula. Well, it is, but the data that we are training it here is, um, with quite large numbers. Neural networks train best if the numbers for the For the input and the output are between the range. For example minus two and plus two or something like this. And this makes it very, very difficult for the neural network to capture this data or for the neuron here.
for i in range(0, 100000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    if i % 100 == 0: 
        print(model.bias)
        print(model.weight)

y1_pred = model(X1)
print("y1_pred =", y1_pred)