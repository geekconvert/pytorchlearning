import pandas as pd

# TfidfVectorizer usually leads to better results because it takes into account that if a word is just, let's say a filler word like the, it has no meaning. And it also takes into account that if a text is longer, the words there also then have less meaning.
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df['spam'] = df['type'] == "spam"
df.drop("type", axis = 1, inplace=True)

vectorizer = TfidfVectorizer(max_features=1000)
messages = vectorizer.fit_transform(df["message"])
print(messages[0, :])
print(vectorizer.get_feature_names_out()[888])
print(vectorizer.get_feature_names_out()[349])


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer

# df = pd.read_csv("./data/SMSSpamCollection", 
#                  sep="\t", 
#                  names=["type", "message"])

# df["spam"] = df["type"] == "spam"
# df.drop("type", axis=1, inplace=True)

# vectorizer = TfidfVectorizer(max_features=1000)
# messages = vectorizer.fit_transform(df["message"])
# print(messages[0, :])
# print(vectorizer.get_feature_names_out()[888])