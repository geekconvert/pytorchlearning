import torch
from torch import nn
import pandas as pd

df = pd.read_csv("./data/student_exam_data.csv")

X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values, 
    dtype=torch.float32
)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1, 1))

hidden_model = nn.Linear(2, 10)
output_model = nn.Linear(10, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Then we can see here that this one is here, this generator object and a generator in Python pretty much just means that we could use a for loop to go to go over them, and then we would get all the parameters here.
print(hidden_model.parameters())
for param in hidden_model.parameters():
    print(param)
# specify the optimizer here to train our models.
# We need to tell this optimizer to not just train our parameters here of the hidden model, but also the ones of the output model.
parameters = list(hidden_model.parameters()) + list(output_model.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.005)

# we now need to first apply here the first hidden layer. Then We will then need to apply the sigmoid function on the output of the hidden layer. And on that output we will then need to apply the output layer or the output model.
# If training a neuron with 500,000 iterations takes too long, try reducing the number of iterations to 250,000 and increasing the learning rate, for example, to 0.025.
# When working with networks, it is possible to get stuck in local minima for several iterations or more due to the random initialization of weights and biases. If the loss does not decrease for a significant portion of the iterations, rerunning the model might help. This issue depends on factors such as the number of training iterations, the learning rate, activation functions and the optimizer. To resolve it, experiment with these parameters until a working solution is found.
# If you are not satisfied with the accuracy and observe that the loss is still decreasing, you can train the model for more iterations. The exact number will depend on how long you are willing to train and the level of accuracy you aim to achieve.
for i in range(0, 600000):
    optimizer.zero_grad()
    # hidden model to be applied to the input data.
    outputs = hidden_model(X)
    # now we need to apply the sigmoid function to output.
    outputs = nn.functional.sigmoid(outputs)
    # now we need to apply our output model here to our existing outputs.
    outputs = output_model(outputs)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 10000 == 0:
        print(loss)


# import sys
# import torch
# from torch import nn
# import pandas as pd

# df = pd.read_csv("./data/student_exam_data.csv")

# X = torch.tensor(
#     df[["Study Hours", "Previous Exam Score"]].values, 
#     dtype=torch.float32
# )
# y = torch.tensor(df["Pass/Fail"], dtype=torch.float32)\
#     .reshape((-1, 1))

# hidden_model = nn.Linear(2, 10)
# output_model = nn.Linear(10, 1)
# loss_fn = torch.nn.BCEWithLogitsLoss()
# parameters = list(hidden_model.parameters()) + list(output_model.parameters())
# optimizer = torch.optim.SGD(parameters, lr=0.005)

# for i in range(0, 500000):
#     optimizer.zero_grad()
#     outputs = hidden_model(X)
#     outputs = nn.functional.sigmoid(outputs)
#     outputs = output_model(outputs)
#     loss = loss_fn(outputs, y)
#     loss.backward()
#     optimizer.step()

#     if i % 10000 == 0:
#         print(loss)

# sys.exit()

# #model.eval()
# #with torch.no_grad():
# #    y_pred = nn.functional.sigmoid(model(X)) > 0.5
# #    y_pred_correct = y_pred.type(torch.float32) == y
# #    print(y_pred_correct.type(torch.float32).mean())
