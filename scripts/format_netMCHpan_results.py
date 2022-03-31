#%%
import os
import sys
import pandas as pd
# %%
path_to_netmhcpan_results = "../data/netMHCpan_results_multi_assay"
results = []
for file in os.listdir(path_to_netmhcpan_results):
    path_to_file = os.path.join(path_to_netmhcpan_results,file)

    with open(path_to_file,"r") as f:
        for line in f:
            line = line.split()
            if len(line) > 1:
                if line[0] == "1":
                    peptide = line[2]
                    HLA = line[1]
                    binding_score = line[11]
                    results.append([peptide,HLA,binding_score])
# %%
binding_results = pd.DataFrame.from_records(results,columns=["peptide","HLA","binding_score"])
# %%
# %%
path_to_dataset = "../data/filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay.csv"
data_set = pd.read_csv(path_to_dataset,index_col=0)
binding_scores = []
for i,row in data_set.iterrows():
    peptide = row["peptide"]
    HLA = row["HLA_allele"]
    binding_row = binding_results[(binding_results['peptide'] == peptide) & (binding_results['HLA'] == HLA)]
    assert len(binding_row) == 1
    binding_scores.append(binding_row['binding_score'].values[0])
data_set['binding_score'] = binding_scores
# %%
data_set.to_csv("../data/filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay_w_binding.csv")
# %%
