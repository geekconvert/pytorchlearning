# TF IDF

- Term frequency : This term frequency would look at a single message. And the more often a word occurs there, the higher this term frequency is. The more words we have in that document, the more we are dividing it by. So this term frequency is kind of like normalizing that so that if, for example, we have a short message, each of the words there have a higher weight, whereas if we, for example, have a longer message, then each of the terms has a lower weight, because then we have more terms in that document that we are dividing by. So this is the term frequency.
- Inverse document frequency : Then we also got the IDF, the inverse document frequency. And there the idea is that we want to reduce the weight of common terms. For example if "the" word the would be contained in all of our messages, then we would take the total number of documents. Let's say for example, we had 5000 documents and 5000 of these documents would contain the word "the". So the word the would be completely meaningless. In that case we would have 5000 divided by 5000. That would be one And then the logarithm of that would be zero.

![TF-IDF visualization](readmeimages/tf_idf.png)
