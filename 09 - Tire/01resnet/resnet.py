# import sys
# import torch
# import torchvision

# from PIL import Image
# from torchvision import transforms

import torch
import torchvision
from PIL import Image
from torchvision import transforms

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.mps.is_available():
#     device = torch.device("mps")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

print(f"Using device: {device}")


# # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
# model.eval()


# pretrained=True means we want to download the model with pre-trained weights.
# We get a warning using torch.hub.load, so we can use torchvision.models instead.
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# weights= means we want to download the model with pre-trained weights.
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# filename = "cat.jpg"
# input_image = Image.open(filename)

filename = "cat.jpg"
input_image = Image.open(filename) #Python image object

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# Pre processing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Input tensor shape: torch.Size([3, 224, 224])
# Input batch shape: torch.Size([1, 3, 224, 224])
input_tensor = preprocess(input_image)
print(f"Input tensor shape: {input_tensor.shape}")
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print(f"Input batch shape: {input_batch.shape}")


# input_batch = input_batch.to(device)
# model.to(device)

input_batch = input_batch.to(device)
model.to(device)

# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# # print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# # print(probabilities)


with torch.no_grad():
    output = model(input_batch) # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    print(f"Output shape: {output.shape}")

probabilities = torch.nn.functional.softmax(output[0], dim=0) # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(f"Probabilities shape: {probabilities.shape}") 
# Output shape: torch.Size([1, 1000])
# Probabilities shape: torch.Size([1000])

# # Read the categories
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
# # Show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())


# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    # print("categories: ", categories)

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], " :: ",  top5_prob[i].item())