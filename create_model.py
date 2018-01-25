import argparse
import os
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("train_dir", help="Path to directory with training data in json format")
parser.add_argument("--model_dir", default="./", help="Path to output directory for model")
parser.add_argument("--model_file", default="model.pkl", help="Filename for model")
args = parser.parse_args()

train_dir = args.train_dir
output_file = os.path.join(args.model_dir, args.model_file)
m = Model()
m.train(train_dir)
m.save(output_file)
