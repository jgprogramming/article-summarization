from tkinter import *
from model import Model
import argparse
import os
from tkinter import messagebox

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./models/", help="Path to output directory for model")
parser.add_argument("--model_file", default="model.pkl", help="Filename for model")
args = parser.parse_args()

model_file = os.path.join(args.model_dir, args.model_file)
m = Model()
m.load(model_file)

root = Tk()

def summariseCallBack():
   tit = title.get()
   tex = text.get("1.0", END)
   summary = m.summarise_text(tex, tit, 5)
   out.delete("1.0", END)
   out.insert("1.0", summary)
   cat = m.predict_category(tex)
   messagebox.showinfo("Category", cat)

Button(root, text ="summarise", command = summariseCallBack).grid(row=0, column=1)

title = Entry(root)
title.grid(row=0, column=0)

text = Text(root)
text.grid(row=1, column=0)

out = Text(root)
out.grid(row=1, column=1)

root.mainloop()
