#author : @akash

import nltk
from nltk.stem import PorterStemmer


stemmer=PorterStemmer()

words=["running", "runs" ,"runner", "easily", "fairly", "lying"]

for word in words:
    stemmed_word=stemmer.stem(word)
    print(f"{word}->{stemmed_word}")


