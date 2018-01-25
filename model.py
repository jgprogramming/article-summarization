import os
import json
from tokenizer import tokenize
from tokenizer import tokenize_sent
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import math
import sys
import numpy as np

class Model:

    categories = None
    corpus_size = 0
    model = None
    position_scores = [0.17, 0.23, 0.14, 0.08, 0.05, 0.04, 0.06, 0.04, 0.04, 0.15]
    heading_weight = 0.3

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
        self.categories = pickle.load(inp)
        self.corpus_size = pickle.load(inp)
        self.model = pickle.load(inp)
        inp.close()

    def summarise_text(self, text, title, top=5):
        cat = self.predict_category(text)
        return(self.summarise_text_cat(text, title, cat, top))


    def summarise_text_cat(self, text, title, category, top):
        scored = self.score_sentences(text, title, category)
        scored = sorted(scored,key=lambda x: x[1], reverse=True)[:top]
        return " ".join(np.array(sorted(scored,key=lambda x: x[0]))[...,2])

    def predict_category(self, text):
        value = -sys.maxsize -1
        cat = None

        for category in self.categories:
            prob = self.calculate_prob(text, category)
            if  prob > value:
                value = prob
                cat = category
        return cat

    def score_sentence(self, sentence, title, category, position):
        features = self.model[category]["features"]
        count = self.model[category]["bow"].transform([sentence])
        tfidf = self.model[category]["tfidf"].transform(count).toarray()[0]
        score = 0
        tokenized_sentence = tokenize(sentence)
        tokenized_title = tokenize(title)
        for word in tokenized_sentence:
            try:
                indx = features.index(word)
                score += tfidf[indx]
            except:
                score += 0

        title_inter = np.intersect1d(tokenized_title, tokenized_sentence)
        score += self.heading_weight * (len(title_inter) / len(tokenized_title))
        return score * self.position_scores[position]

    def score_sentences(self, text, title, category):
        sentences = tokenize_sent(text, title)
        out = []
        for indx, sentence in enumerate(sentences):
            score = self.score_sentence(sentence, title, category, int(10*indx/len(sentences)))
            out.append([indx, score, sentence])
        return out

    def calculate_prob(self, text, category):
        prob = math.log(self.model[category]["prob"])

        features = self.model[category]["features"]
        count = self.model[category]["count"]
        suma = self.model[category]["sum"]

        for word in tokenize(text):
            try:
                indx = features.index(word)
                c = count[indx]
            except:
                c = 1
            prob += math.log(c/suma)
        return prob
