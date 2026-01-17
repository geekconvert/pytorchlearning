# import torch

# weights = torch.load("fc_model_3.pth", weights_only=True, map_location="cpu")
# torch.save(weights, "fc_model_3.pth")
# print(weights)

import torch

# weights = torch.load("fc_model_3.pth", weights_only = True)

# # this print device='mps:0' along with the data. The reason is that they have been stored like this in a sense that these weights come originally from an MPS device. However, if I were now to execute this somewhere else, this code could lead to problems, especially on systems where the MPs device is not available.
# print(weights)


# Now here the weights here have been stored on the CPU.
weights = torch.load("fc_model_3.pth", weights_only = True, map_location="cpu")
# Now default will be set to CPU
torch.save(weights, "fc_model_3.pth")
print(weights)