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
