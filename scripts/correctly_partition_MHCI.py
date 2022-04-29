#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# %%
all_data = pd.read_csv("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/filtered_data_IEDB_4_tested_len_9_10_full_HLA_IFNg_assay_w_parts.csv",index_col=0)
fig = plt.figure(figsize=(20,10))
unique_HLAS = list(set(all_data["HLA_allele"]))

all_data_fixed = pd.DataFrame()

for i,HLA in enumerate(unique_HLAS):
    immunogenic_subjects = all_data[(all_data["HLA_allele"] == HLA) & (all_data["immunogenicity"] == "Positive")].sort_values(by="positive_subjects")
    non_immunogenic_subjects = all_data[(all_data["HLA_allele"] == HLA) & (all_data["immunogenicity"] == "Negative")].sort_values(by="tested_subjects")
    max_len = min(len(immunogenic_subjects),len(non_immunogenic_subjects))
    all_data_fixed = pd.concat((all_data_fixed,immunogenic_subjects.iloc[:max_len,:],non_immunogenic_subjects.iloc[:max_len,:]),axis=0)


# %%
all_data_fixed = all_data_fixed.sample(frac=1, random_state=1).reset_index(drop=True)
all_data_fixed.shape
all_data_fixed.to_csv("../data/ifng_test.csv")

# %%
partioning_format = pd.DataFrame()
partioning_format["peptide"] = all_data_fixed["peptide"]
partioning_format["target"] = [1 if x == "Positive" else 0 for x in all_data_fixed["immunogenicity"]]
partioning_format["HLA_allele"] = all_data_fixed["HLA_allele"]
partioning_format.to_csv("../data/ifng_test_partioning.tsv",sep=" ",index=False)
# %%

# %%
