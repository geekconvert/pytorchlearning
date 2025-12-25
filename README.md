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

![Neuron to Neural network](readmeimages/neuron_to_neuron_network1.png)

![Neuron to Neural network](readmeimages/neuron_to_neuron_network2.png)
