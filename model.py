import os
import json
from tokenizer import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

class Model:

    categories = None
    corpus_size = 0
    model = None
    def __init__(self):
        pass

    def train(self, training_dir):
        self.categories = os.listdir(training_dir)

        data = {}
        for category in self.categories:
            data[category] = []

            for file in os.listdir(os.path.join(training_dir, category)):
                doc_path = os.path.join(training_dir, category, file)
                d = open(doc_path)
                j = json.loads(d.read())
                for document in j:
                    doc = {"title":document["title"], "text":document["text"]}
                    data[category].append(doc)

            self.corpus_size += len(data[category])

        self.model = {}

        for category in self.categories:
            self.model[category] = {}
            self.model[category]["prob"] = len(data[category])/self.corpus_size
            bow = CountVectorizer(input='content', tokenizer=tokenize, stop_words='english')
            korpus = []
            for entry in data[category]:
                korpus.append(entry["text"])
            bow = bow.fit(korpus)
            transformed_korpus = bow.transform(korpus)
            count = [ sum(x) for x in zip(*transformed_korpus.toarray()) ]

            tfidf = TfidfTransformer(norm="l2")
            tfidf.fit(transformed_korpus)

            self.model[category]["bow"] = bow
            self.model[category]["tfidf"] = tfidf
            self.model[category]["count"] = count
            self.model[category]["features"] = bow.get_feature_names()
            self.model[category]["sum"] = sum(self.model[category]["count"])

    def save(self, output):
        out = open(output, 'wb')
        pickle.dump(self.categories, out)
        pickle.dump(self.corpus_size, out)
        pickle.dump(self.model, out)
        out.close()

    def load(self, input):
        inp = open(input, 'rb')
        self.model = pickle.load(inp)
        self.corpus_size = pickle.load(inp)
        self.categories = pickle.load(inp)
        inp.close()
