# %%
import pandas as pd
import numpy as np
import os
import sys
# %%
df = pd.read_csv("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/df_before_binding_cores.csv",index_col=0)
# %%
path_to_MHC = "/Users/christianpederjacobsen/tools/netMHCIIpan-4.0/netMHCIIpan"
unique_MHCs = sorted(set(df["MHCs"]))
for MHC in unique_MHCs:
    subsample = df[df["MHCs"] == MHC]
    unique_peptides = set(subsample["Peptide"])
    path_to_peptide_file = f"../data/MHCII_netMHCII_in/{MHC}.pep"
    pep_file = open(path_to_peptide_file,"w+")
    print("\n".join([f"{x}\t-1" for x in unique_peptides]),file=pep_file)
    pep_file.close()
    outfile_name = f"../data/MHCII_netMHCII_out/{MHC}.out"
    if len(MHC) > 10:
        MHC = "HLA-" + MHC
    # print(f"{path_to_MHC} -inptype 1 -f {path_to_peptide_file} -a {MHC} -filter > {outfile_name}")
    os.system(f"{path_to_MHC} -inptype 1 -f {path_to_peptide_file} -a {MHC} > {outfile_name} &")

# %%
