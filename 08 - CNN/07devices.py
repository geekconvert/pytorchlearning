import torch 
from torch import nn

# device = torch.device("cpu122") # RuntimeError: Invalid device string: 'cpu122'

device = torch.device("cpu") # by default I can run my code on CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print("Running on device: :", device)

a = torch.tensor(4)
b = torch.tensor(5)

# These tensors were actually stored on our CPU and this has been run on our CPU. The reason is that by default everything here is created on our CPU. So this is now here running on our CPU.
print(a.device)
print(b.device)

print(a + b)

######################

a = torch.tensor(4, device=device)
b = torch.tensor(5)

print(a.device) #mps
print(b.device) #cpu

print(a+b) #mps # this is only the case because we have a single value here.


########################

a = torch.tensor([5, 7], device=device)
b = torch.tensor([3, 4])

#print(a+b) # this will give an error because a is on mps and b is on cpu. Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!


########################

a = torch.tensor([5, 7], device=device)
b = torch.tensor([3, 4])

b=b.to(device)

print(a+b)