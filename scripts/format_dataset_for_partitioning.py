import os
import pandas as pd
import sys
import numpy as np

dataset = pd.read_csv("../data/deep_immuno_2.csv")
dataset["target"] = [1 if "Positive" in x else 0 for x in dataset["immunogenicity"]]
dataset[["peptide","target","HLA"]].to_csv("../data/partioning_format_deepimmuno.tsv",sep="\t",index=False)