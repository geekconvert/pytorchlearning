import torch
from torch import nn

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Running on device:", device)

a = torch.tensor([5, 7], device=device)
b = torch.tensor([3, 4], device=device)

print(a.shape) #torch.Size([2])
a = a.reshape(1, 2) # And this is here the input that our model now expects.

model = torch.nn.Sequential(
    nn.Linear(2, 10)
)

# outputs = model(a) # RuntimeError: MPS device does not support linear for non-float inputs

a = torch.tensor([5, 7], device=device, dtype=torch.float32)
a = a.reshape(1, 2)

#outputs = model(a) # RuntimeError: Tensor for argument weight is on cpu but expected on mps. Reason: weights here are stored on the CPU and the data here is on MPS.

model = torch.nn.Sequential(
    nn.Linear(2, 10)
).to(device)

outputs = model(a)
print("outputs: ", outputs)

probabilities = nn.functional.softmax(outputs, dim=1)
print("probabilities: ", probabilities) # still on mps only
print(probabilities[0, 0]) # still on mps only
print(probabilities[0, 0].item())
print(probabilities[0, :]) # mps
print(probabilities[0, :].tolist()) # cpu

# So of course in order to print out everything here to the terminal the data needs to be transferred from the device. And also of course if you are turning something into a simple, let's say Python list, then also the data has to be transferred to the CPU. But aside from that, the whole code here can run on the GPU and does not, or the data does not always get transferred to the CPU.