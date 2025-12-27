# TF IDF

- Term frequency : This term frequency would look at a single message. And the more often a word occurs there, the higher this term frequency is. The more words we have in that document, the more we are dividing it by. So this term frequency is kind of like normalizing that so that if, for example, we have a short message, each of the words there have a higher weight, whereas if we, for example, have a longer message, then each of the terms has a lower weight, because then we have more terms in that document that we are dividing by. So this is the term frequency.
- Inverse document frequency : Then we also got the IDF, the inverse document frequency. And there the idea is that we want to reduce the weight of common terms. For example if "the" word the would be contained in all of our messages, then we would take the total number of documents. Let's say for example, we had 5000 documents and 5000 of these documents would contain the word "the". So the word the would be completely meaningless. In that case we would have 5000 divided by 5000. That would be one And then the logarithm of that would be zero.

![TF-IDF visualization](readmeimages/tf_idf.png)

# Sigmoid function

- function that can just map linear output to the range of 0 to 1.

![sigmoid function](readmeimages/sigmoid.png)

# Using sigmoid and BCEWithLogitsLoss

- Using we got a little bit of a challenge there when we are using the MSE loss. The problem is that the error there is just too small. And then the gradients become small as well. And then the neuron struggles to learn efficiently.
- So yes, we apply the sigmoid function to the output of the neuron. But then when it comes to the loss, we use this BCE loss (binary cross entropy loss) to kind of like undo a little bit of the effects of the sigmoid function. And then we still have this pulling force when it comes to the loss function.
- So we are applying the sigmoid so that we get probabilities to the neuron. And then with the BCE loss function, we are then kind of like undoing a little bit there so that we can still train properly,

![sigmoid function](readmeimages/using_sigmoid.png)

# Evaluating model

## Accuracy:

- This would just answer the question for what percentage is the true value of our messages, whether they are spam or not. Spam exactly identical to the prediction predicted value.

## sensitivity:

- sensitivity or true positive rate: This is the proportion of actual spam messages that got correctly identified. gGiven that we got a spam message, what percentage did the model predict correctly.

- It may be accurate in a sense that It detects a lot of things correctly, but for example, the sensitivity could be really low and in this case we would not actually filter many spam messages.

- For example, let's say you got you get 1000 normal emails and one spam email. If you had a model that would always accept all the emails and never reject any emails, then the accuracy would be really high because we have 1000 messages that got correctly accepted and one spam message that just got accepted as well However, the sensitivity here would be extremely low. It would be zero here, because the proportion of actual spam messages that got correctly identified

## specificity or the true negative rate:

- This is the proportion of actual non-spam messages that got correctly identified. So given that we are not having a spam message, what percentage did the model predict correctly.

- if you don't get an email from a potential customer and that would be a complaint, that could have serious consequences.

## Prediction:

- This stands for the proportion of predicted spam messages that are actually spam.

- given that the that the prediction is spam, what percentage had been spam originally.

![evaluating model](readmeimages/evaluating_model.png)

# Train / Validation / Test:

- we got normal parameters here(like bias, weights etc) that our model has. These are learned here during training.

- But there are hyper parameters that influence the structure of the model. For example for the countvectorizer how many features we got there, how many inputs we have here, what's the learning rate, how many training passes we got, where we do the cut off here when we consider something spam or not spam. So all of these things would be considered hyper parameters here.

![evaluating model](readmeimages/training_validation_test.png)

# idea Embeddings

- And then the question would be, can't we just use this then to enhance or to run our spam filter. And it turns out that if we do so, then our spam filter will suddenly work extremely well because we are then no longer training it on individual words, but on the meaning of the text.
- And then suddenly also completely unknown things that, for example, just nowadays would be used inside emails could be captured as spam.

![Idea embeddings](readmeimages/idea_embeddings.png)

# Neuron to Neural network

- So, far we have learned how we can train a single neuron and how that one can learn. The next step is that we want to stack multiple neurons together to create a neural network.

- We had a single neuron that made a prediction here based on some inputs. And what we are going to explore throughout this chapter is what if we, for example, just put a few neurons before they can do some processing here. And then the output here then gets used as an input for this neuron. And then we use this to make the prediction.

- So you can see here how now we just have a stack of neurons here. They all have access to all the inputs here in this case. So all the inputs are connected to all the neurons here. They all have individual weights and also biases. They make here an intermediate prediction. And then this is being used for this neuron to then make the final prediction here.

![Neuron to Neural network](readmeimages/neuron_to_neuron_network1.png)

![Neuron to Neural network](readmeimages/neuron_to_neuron_network2.png)

# Linearity problem and non-linear activation functions

- We want to explore in a visual way why we actually need a neural network, why a single neuron can't be able to grab all the problems, or why we can't use it to solve all the problems.

- So linearity means that we have limited learning capabilities. We are not going to be able to capture complex non-linear patterns in our data.

- A single linear neuron, even with a sigmoid function, won't be able to capture these non-linear relationships.

- For example, let's say we just had two variables, and based on that, we needed to make a prediction. Blue or orange. And in that case here in this case this would be a simple relationship that could be learned.

- Because what we could do is that we could just kind of like draw the line like this and then we would be able to capture this here.

- Um, so this would be a relationship that we would be able to capture with our linear neuron, and then we would put the sigmoid activation function on it, uh, to then we would have something like this here. Um, for example, orange could then be negative values. Blue could then be positive values. And then they get turned through the sigmoid function into probabilities.

- Well we could try we could try something like this. Well it's not going to be able to describe the data.

- What we would need now would be something like this that we would be like, okay, all the blue entries are inside this area here, and this now would just be a relationship that we would not be able to capture with a linear neuron.

- So linearity means that we got limited learning power. And this also means if we would now just have a network here that if the hidden layers are linear, the final output layer would also be just linear.

![Linearity problem](readmeimages/linearity_problem.png)

![Non linear activation functions](readmeimages/non_linear_activation_function.png)

# BackPropagation

![BackPropagation](readmeimages/backpropagation.png)

- Math behind BackPropagation

- The error is kind of like just the meaning of how would we need to have changed this output here with respect to our loss function

![back_propagation Math](readmeimages/back_propagation.png)

# Structure of the network

- We've chosen a hidden layer size of ten neurons. Well I had to choose something. So this felt like could be something that could work, but this would be something that we would need to tune with by withholding validation data. And this is something that we would need to tune just as all the other hyperparameters in our network. But I would say ten neurons in the hidden layer could be a good starting point.

- We we got ten hidden outputs here. They are all then connected to the last neuron which is then going to make a make a prediction. And of course there the sigmoid is applied. The sigmoid function is being applied to that because we are using the BCEWithLogitsLoss Loss function that includes the sigmoid function there.

![structure_of_the_network](readmeimages/structure_of_the_network.png)

# ReLU

- So far here we had our inputs and then we had ten neurons here in this layer. It's abbreviated here again. But we calculated the sum and then applied the sigmoid function to it. And then fed this output into one last neuron to then actually make the prediction. So we used the sigmoid function as an activation function in between just because it breaks linearity.

- So once this linearity has had been broken, the weights of the network could then adjust to the rest

- in ReLU all the values below zero turn into just zero, and then all the values above zero. They are just being taken as they are.

- the sigmoid function outputted values between 0 and 1. And they can result into very very small gradients during back propagation. So if we had multiple layers there The gradients could become small, and then it's more difficult for the network to adjust the weights. However, for us here so far our network was relatively small, so this was not a major issue.

- Um, the the advantage here of the ReLU function is that it outputs either zero or the input itself if positive. And this is extremely easy, then to take the gradient. Fr example here, if it would be one, then we would just need to divide it by one, which we don't even need to do. We can just backpropagate it directly. And if we are at less than 0, then the activation there with respect to this neuron would be zero. And in that case we can just ignore it. So this makes backpropagation extremely simple because either the gradient here is exactly one or it is zero.

- Also we don't have, gradients that become smaller and smaller. Here the gradient would be one or zero. This property would be the so-called sparse activation.

- And just imagine we would be using this on the output of ten neurons. Then probably by chance, about five of these neurons would output a value below zero, which would then be just after applying ReLU. It would just be a zero then. And five of the neurons would predict a value above zero. And then we would just take the value here. So when we are using this activation function, only about 50% of neurons are activated. And this reduced or reduces the computational complexity and improves the generalization of the network, meaning that if it can't rely on some neurons or they suddenly are just zero, then the network is forced to look at the big picture instead of getting lost in the details.

- The next thing is that, of course, everything here is quite efficient to compute because it only involves a max operation which is extremely efficient or easy to calculate. And this is significantly easier to compute compared to the exponentials that we had when we utilized the sigmoid functions.

- Last but not least, this function of course, also breaks linearity because this is no longer a linear equation here if we have a look at this. So linearity has also been broken here.

![relu_1](readmeimages/relu_1.png)

![relu_2](readmeimages/relu_2.png)

![relu_3](readmeimages/relu_3.png)

![relu_4](readmeimages/relu_4.png)

# Adam Optimizer

![adam_optimizer](readmeimages/adam_optimizer.png)
