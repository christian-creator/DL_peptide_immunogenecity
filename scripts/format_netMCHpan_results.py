#%%
import os
import sys
import pandas as pd
# %%
path_to_netmhcpan_results = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/netMHCpan_results_multi_assay"
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
binding_results.to_csv("data/binding_results_multi_assay.csv")
# %%
