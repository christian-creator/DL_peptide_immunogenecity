import os
import pandas as pd
import sys
import numpy as np

dataset = pd.read_csv("data/filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay.csv")
dataset["target"] = [1 if "Positive" in x else 0 for x in dataset["Response"]]
dataset[["peptide","target","HLA_allele"]].to_csv("data/partioning_format_filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay.tsv",sep="\t",index=False)