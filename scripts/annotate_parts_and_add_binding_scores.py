# %%
import pandas as pd
import numpy as np
import os
import sys
all_data = pd.read_csv("../data/ifng_test.csv",index_col=0)
binding_scores = pd.read_csv("../data/binding_results_ifng_2.csv",index_col=0)
# %%
# add parts column with zeros all the way 
n = all_data.shape[0]
empty_column = np.zeros((n))
all_data["parts"] = empty_column
all_data["binding_score"] = empty_column
path_to_parts = "../data/ifng_assay_parts_test"
counter = 0 
for file in os.listdir(path_to_parts):
    part = int(file[-1])
    print("started on new file",part)
    path_to_file = os.path.join(path_to_parts,file)
    partion_data = pd.read_csv(path_to_file,sep=" ",names=["peptide","label","HLA_allele"])
    for i, row_partition in partion_data.iterrows():
        peptide_partition = row_partition["peptide"]
        hla_partition = row_partition["HLA_allele"]
        try:
            binding_score = binding_scores[(binding_scores["peptide"] == peptide_partition) & (binding_scores["HLA"] == hla_partition)]["binding_score"].values[0]
        except IndexError:
            print(peptide_partition,hla_partition)
            counter += 1
        for j, row_all_data in all_data.iterrows():
            if row_all_data["peptide"] == peptide_partition and row_all_data["HLA_allele"] == hla_partition:
                all_data.loc[j,"parts"] = int(part)
                all_data.loc[j,"binding_score"] = binding_score
                # print(all_data.loc[j])
                # print(row_all_data)
                # print(part)
print(counter)

# %%

# %%
all_data.to_csv("../data/ifng_test_w_parts_w_binding_scores.csv")
# %%
