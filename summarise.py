import argparse
import os
import json
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input in json format with title and text fields")
parser.add_argument("--model_dir", default="./models/", help="Path to output directory for model")
parser.add_argument("--model_file", default="model.pkl", help="Filename for model")
parser.add_argument("--top", default="5", help="How many sentences to output")
parser.add_argument("--debug", default=False, help="Print debug messages")
args = parser.parse_args()

inputFile = args.input
model_file = os.path.join(args.model_dir, args.model_file)
top = int(args.top)
debug = bool(args.debug)

m = Model()
m.load(model_file)
d = open(inputFile)
j = json.loads(d.read())



title = j[0]["title"]
text = j[0]["text"]
summary = m.summarise_text(text, title, top)

print(title)
print(summary)

if debug == True:
    print("predicted category: ", m.predict_category(text))
    print("Shortened to ", len(summary) / len(text)*100, "% of oryginal text")
