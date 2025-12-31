import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)
print(df.head())


# max_features limits the number of features to the most frequent words
cv = CountVectorizer(max_features=3)
documents = ['Hello world, Today is amazing','Hello mars, today is perfect']
cv.fit(documents)
# get_feature_names_out() returns the list of feature names (words) learned by the CountVectorizer
print("cv feature names",cv.get_feature_names_out())


# Be aware sometimes this countvectorizer also has some pre-processing attached to it. like "Today" and "today" are considered same. "Hello!" is considered as "Hello"
documents = ['Hello! world, Today is amazing','Hello mars, today is perfect']
# So fitting just means that the countvectorizer fits or learns which words to look at.
cv.fit(documents)

# this will then just tell us which features this Countvectorizer has learned or is going to keep a look at. It analyzes all the documents here and then it takes the three words that occur the most often here.
print("cv feature names",cv.get_feature_names_out())

cv= CountVectorizer(max_features=6)
documents = [
    "Hello world. Today is amazing. Hello hello",
    "Hello mars, today is perfect"
]
cv.fit(documents)
out = cv.transform(documents)
# Then we can see here that this one here is a compressed sparse row matrix. This is a special type of data structure. The idea is that from a mathematical point we have a matrix. But because there are so many zeros usually in this matrix, um, it doesn't even make sense to print out all the zeros. So what happens is that only the coordinates here where values are being Used are being printed here, and this is the way how the matrix is being outputted here.
print("out: ",out)
print("cv feature names",cv.get_feature_names_out())
# but we can turn this into a normal matrix by just saying here .todense(). This will tell us how many times each of the words appeared in each of the documents. like "amazing" appeared once in the first document and zero times in the second document.
print("out.todense(): ",out.todense())


cv = CountVectorizer(max_features=1000)
# meaning learning the 1000 words to look at and also transforming the data both at the same time.
messages = cv.fit_transform(df['message'])
print("messages: ",messages)
print("messages[0, :1]: ",messages[0, :1])
print(cv.get_feature_names_out()[888])
print(cv.get_feature_names_out()[349])


# from sklearn.feature_extraction.text import CountVectorizer

# df = pd.read_csv("./data/SMSSpamCollection", 
#                  sep="\t", 
#                  names=["type", "message"])

# df["spam"] = df["type"] == "spam"
# df.drop("type", axis=1, inplace=True)

# cv = CountVectorizer(max_features=1000)
# messages = cv.fit_transform(df["message"])
# print(messages[0, :])
# print(cv.get_feature_names_out()[888])

# ----
# cv = CountVectorizer(max_features=6)
# documents = [
#     "Hello world. Today is amazing. Hello hello",
#     "Hello mars, today is perfect"
# ]
# cv.fit(documents)
# print(cv.get_feature_names_out())
# out = cv.transform(documents)
# print(out.todense())