import pandas as pd
import numpy as np

data = pd.read_csv("data/filtered_data_IEDB_4_tested_len_9_10_full_HLA.csv")
sample_allele = data[data["HLA_allele"]=="HLA-A*02:01"]
print(sample_allele)
# Creating sample of data
# allele_data = data[data[""]==]