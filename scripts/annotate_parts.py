import pandas as pd
import numpy as np
import os

big_file = pd.read_csv("../data/filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay_w_binding.csv")

# add parts column with zeros all the way 
n = big_file.shape[0]
empty_column = np.empty((n))
big_file = big_file.assign(parts=empty_column)

for file in os.listdir("../data/multi_assay_parts"):
    print("started on new file")
    part = int(file[-1])
    path_to_file = os.path.join("../data/multi_assay_parts",file)
    data = pd.read_csv(path_to_file,sep="\t",names=["peptide","label","HLA_allele"])
    for i in range(data.shape[0]):
        small_file_peptide = data.loc[i, "peptide"]
        small_file_hla = data.loc[i, "HLA_allele"]
        for j in range(big_file.shape[0]):
            peptide = big_file.loc[j,"peptide"]
            mhc = big_file.loc[j,"HLA_allele"]
            if peptide == small_file_peptide and mhc == small_file_hla:                
                big_file.iloc[j,"parts"] = part
                break
  
print(big_file.loc[133,"parts"])
big_file.to_csv("../data/astrids_bigfile")