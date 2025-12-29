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

# this is the model we want to train
model = nn.Linear(1, 1)

# now we want to prepare for training. And what we want to do here is that we first need to specify a few things.
# loss function is mean squared error loss. And this loss function just happens to be already implemented in pytorch.
loss_fn = torch.nn.MSELoss()

# if loss function needs to added used manually, we can do it like this:
print("loss_fn", loss_fn(
    torch.tensor([5.5], dtype=torch.float32), 
    torch.tensor([10.0], dtype=torch.float32)
))
# we also need to configure how the model should be optimized. Stochastic Gradient Descent. This stochastic gradient descent or all the optimizers in PyTorch, they expect us as a first parameter to pass in the parameters that can be optimized here. And only these parameters are going to get optimized.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

print(model.weight)
print(model.bias)

# Training pass
optimizer.zero_grad() # This kind of like just tells the optimizer, hey, start from scratch. Now we are now doing a new pass here. Every time we are doing a new pass we need to tell the optimizer, hey, now a new pass is going to come and then the optimizer can, um, discard some data that it had collected before.
outputs = model(X1)
loss = loss_fn(outputs, y1)
# This is now here calculating the gradients meaning the slope or the steepness of the curves with respect to all of our parameters. So once we calculated this then it knows how much. For example, the parameter b or w, one would need to be changed.
loss.backward()
# So we now know the gradients, but we haven't changed our parameters yet. Only the optimizer here knows our learning rate. So here we are calculating the gradients in which direction the parameters need to move. And then we tell the optimizer hey, now actually perform an optimizer step.
optimizer.step()

print(model.weight)
print(model.bias)


y1_pred = model(X1)
print("y1_pred =", y1_pred)