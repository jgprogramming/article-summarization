import argparse
import os
from model import Model
import json

parser = argparse.ArgumentParser()
parser.add_argument("test_dir", help="Path to test dir")
parser.add_argument("--model_dir", default="./models/", help="Path to output directory for model")
parser.add_argument("--model_file", default="model.pkl", help="Filename for model")
args = parser.parse_args()

test_dir = args.test_dir
model_file = os.path.join(args.model_dir, args.model_file)
m = Model()
m.load(model_file)

categories = os.listdir(test_dir)

good = 0.
bad = 0.

for category in categories:
    for file in os.listdir(os.path.join(test_dir, category)):
        doc_path = os.path.join(test_dir, category, file)
        d = open(doc_path)
        j = json.loads(d.read())
        for document in j:
            txt = document["text"]
            if category != m.predict_category(txt):
                print(category, "bad")
                bad += 1
            else:
                print(category, "good")
                good += 1

print(good / (bad + good))
