# %%
import pandas as pd
import numpy as np
import os
import sys
# %%
path_to_immunogenecity = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/Therapeutic_ADA.xlsx"
path_to_seqs = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/Therapeutic_Sequences.xlsx"
immunogenecity_df = pd.read_excel(path_to_immunogenecity,skiprows=2).iloc[:,:2]
VH_seqs = pd.read_excel(path_to_seqs,skiprows=3,sheet_name=0)
VL_seqs = pd.read_excel(path_to_seqs,skiprows=3,sheet_name=1)
VL_seqs = VL_seqs.drop('Origin',axis=1)
VH_seqs = VH_seqs.drop('Origin',axis=1)

# %%
immunogenecity_file = open("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/Humab_immunogenecity.txt","w+")
fasta_folder = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/fastas"
counter = 0
for index, theraputic in immunogenecity_df.iterrows():
    name = theraputic["Name"]
    immunogenecity = theraputic["Immunogenicity"]
    VH = VH_seqs[VH_seqs["Name"] == theraputic["Name"]]
    VL = VL_seqs[VL_seqs["Name"] == theraputic["Name"]]

    if len(VH) > 0 and len(VL) > 0:
        print(f"{name}\t{immunogenecity}",file=immunogenecity_file)
        # Create fasta files
        VH_seq = "".join(VH.iloc[0,1:].to_list())
        VH_seq = VH_seq.replace("-","")
        VL_seq = "".join(VL.iloc[0,1:].to_list())
        VL_seq = VL_seq.replace("-","")

        fasta_file = open(os.path.join(fasta_folder,name) + ".fasta","w+")
        print(f">{name}_HC", file=fasta_file)
        print(VH_seq, file=fasta_file)

        print(f">{name}_LC", file=fasta_file)
        print(VL_seq, file=fasta_file)

immunogenecity_file.close()
# %%
