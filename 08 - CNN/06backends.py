import torch

print("cuda: ", torch.cuda.is_available())

print("MPS: ", torch.mps.is_available())

# import torch

# print("Cuda:", torch.cuda.is_available())
# print("MPS:", torch.mps.is_available())