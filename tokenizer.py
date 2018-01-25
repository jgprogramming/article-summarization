import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    stems = [word.lower() for word in stems if word.isalpha()]
    return stems
