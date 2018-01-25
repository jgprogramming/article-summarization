import argparse
import os
import json
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input in json format with title and text fields")
parser.add_argument("--model_dir", default="./", help="Path to output directory for model")
parser.add_argument("--model_file", default="model.pkl", help="Filename for model")
parser.add_argument("--top", default="5", help="How many sentences to output")
parser.add_argument("--debug", default="false", help="Print debug messages")
args = parser.parse_args()

inputFile = args.input
model_file = os.path.join(args.model_dir, args.model_file)
m = Model()
m.load(model_file)

d = open(inputFile)
j = json.loads(d.read())

print(j["title"])

print(m.summarise_text(j["text"], j["title"]))
