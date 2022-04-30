# %%
import pandas as pd
import numpy as np
import os
# %%
# Reading the dataset
data = pd.read_csv("../data/ifng_test_balanced.csv")
# %%
# Creating a .txt file for each HLA allele with all pepties bound to that allele in the filtered dataset
path_to_peptide_files = "../tmp/peptides_file_ifng_true"
os.system(f"mkdir -p {path_to_peptide_files}")



for allele in set(data["HLA_allele"]):
    file_name = allele + "_peptides_txt"
    path_to_outfile = os.path.join(path_to_peptide_files,file_name)
    print(path_to_outfile)
    outfile = open(path_to_outfile,"w+")
    peptides_for_allele = data[data["HLA_allele"]==allele]
    peptides = list(peptides_for_allele["peptide"])
    print("\n".join(peptides),file=outfile)
    outfile.close()


# %%
