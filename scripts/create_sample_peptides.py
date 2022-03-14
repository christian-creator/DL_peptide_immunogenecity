import pandas as pd
import numpy as np
import os

# Reading the dataset
data = pd.read_csv("data/filtered_data_IEDB_4_tested_len_9_10_full_HLA.csv")

# Creating a .txt file for each HLA allele with all pepties bound to that allele in the filtered dataset
path_to_peptide_files = "tmp/peptides_file"
for allele in data["HLA_allele"]:
    file_name = allele + "_peptides_txt"
    path_to_outfile = os.path.join(path_to_peptide_files,file_name)
    outfile = open(path_to_outfile,"w+")
    peptides_for_allele = data[data["HLA_allele"]==allele]
    peptides = list(peptides_for_allele["peptide"])
    print("\n".join(peptides),file=outfile)

