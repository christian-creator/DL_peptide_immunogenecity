#%%
import pandas as pd
import numpy as np
import os
import sys
#%%
def read_MHCII_results(path_to_folder):
    data = dict()
    for file_name in os.listdir(path_to_folder):
        path_to_file = os.path.join(path_to_folder,file_name)
        with open(path_to_file,"r") as infile:
            HLA = file_name[:-4]
            data[HLA] = dict()
            for line in infile:
                if len(line) > 1:
                    line = line.split()
                    try:
                        int(line[0])
                    except ValueError:
                        continue
                    
                    peptide = line[2]
                    binding_core = line[4]
                    core_reliability = line[5]
                    EL_score = line[8]
                    data[HLA][peptide] = [binding_core,core_reliability,EL_score]
    return data

binding_core_data = read_MHCII_results("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/MHCII_netMHCII_out")
# %%
df = pd.read_csv("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/df_before_binding_cores.csv",index_col=0)
df
# %%
core_data = pd.DataFrame([binding_core_data[row["MHCs"]][row["Peptide"]] for index,row in df.iterrows()],columns=["core","core_rel","rank_score"])
result_df = pd.concat((df.reset_index(drop=True), core_data.reset_index(drop=True)), axis=1)
result_df.to_csv("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/df_after_binding_cores.csv")
# %%
