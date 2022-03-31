# %%
import pandas as pd
import numpy as np
import os
# %%
# Reading the dataset
data = pd.read_csv("data/semi_supervised_data.csv")
# %%
# Creating a .txt file for each HLA allele with all pepties bound to that allele in the filtered dataset
path_to_peptide_files = "tmp/peptides_file_supervised"
for allele in set(data["HLA_allele"]):
    file_name = allele + "_peptides_txt"
    path_to_outfile = os.path.join(path_to_peptide_files,file_name)
    outfile = open(path_to_outfile,"w+")
    peptides_for_allele = data[data["HLA_allele"]==allele]
    peptides = list(peptides_for_allele["peptide"])
    print("\n".join(peptides),file=outfile)
    outfile.close()

