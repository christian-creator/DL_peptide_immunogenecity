#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys


def read_HLA_database(path):
    HLA_dict = dict()
    with open(path,"r") as file:
        for line in file:
            line = line.split()
            HLA = line[0].replace("HLA-","")
            seq = line[1]
            HLA_dict[HLA] = seq
    return HLA_dict

    
#%%
# Read the raw excel files
## Read epitope data
path_to_epitope_file = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/MHCII/epitope_data_MHCII.csv"
epitope_raw_df = pd.read_csv(path_to_epitope_file,skiprows=1)
epitope_raw_df = epitope_raw_df[['Epitope ID',"Description","Antigen Name"]]
## Read experiment data
path_to_experiment_file = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/MHCII/experiment_data_MHCII.csv"
experiment_raw_df = pd.read_csv(path_to_experiment_file,skiprows=1)
experiment_raw_df = experiment_raw_df[['Epitope ID',"Cell Type","Allele Name","Antigen Description","Qualitative Measure","Assay Group","Number of Subjects Tested","Number of Subjects Responded"]].dropna()
experiment_raw_df = experiment_raw_df[experiment_raw_df["Number of Subjects Tested"] > 0]
experiment_raw_df.sort_values(by='Epitope ID', ascending=True)
#%%
# Remove observations with inadequate MHCII annotations
experiment_df = experiment_raw_df[(experiment_raw_df["Allele Name"].str.contains('\*')) & (~experiment_raw_df["Allele Name"].isna())]
#%%
# Filter for lengths of interest
lengths_of_interest = [15]
experiment_df["epitope_length"] = experiment_df["Antigen Description"].apply(lambda x: len(x.strip()))
epitope_raw_df["epitope_length"] = epitope_raw_df["Description"].apply(lambda x: len(x.strip()))
experiment_df = experiment_df[experiment_df.epitope_length.isin(lengths_of_interest)]
epitope_df = epitope_raw_df[epitope_raw_df.epitope_length.isin(lengths_of_interest)]
#%%
# Filter for experiment of interest
experiment_of_interest = ["IFNg release"]
experiment_df = experiment_df[experiment_df["Assay Group"].isin(experiment_of_interest)]

#%%
# Sum over the epitope-MHCII and count the number of positive subjects
eptitope_IDs = []
peptides = []
MHCs = []
number_of_tested_subjects = []
number_of_positive_subjects = []
unique_observations = set()
for index,row in tqdm(experiment_df.iterrows(),total=experiment_df.shape[0]):
    epitope_ID = row["Epitope ID"]
    peptide = row["Antigen Description"]
    MHC = row["Allele Name"]
    peptide_HLA = peptide + MHC
    if peptide_HLA not in unique_observations:
        sub_sample = experiment_df[(experiment_df["Epitope ID"] == epitope_ID) & (experiment_df["Antigen Description"] == peptide) & (experiment_df["Allele Name"] == MHC)]
        number_tested_subjects = np.sum(sub_sample["Number of Subjects Tested"])
        number_positive_subjects = np.sum(sub_sample["Number of Subjects Responded"])
        eptitope_IDs.append(epitope_ID)
        peptides.append(peptide)
        MHCs.append(MHC)
        number_of_tested_subjects.append(number_tested_subjects)
        number_of_positive_subjects.append(number_positive_subjects)
        unique_observations.add(peptide_HLA)

#%%
df = pd.DataFrame({"Epitope ID":eptitope_IDs,
                    "Peptide":peptides,
                    "MHCs":MHCs,
                    "Tested Subjects":number_of_tested_subjects,
                    "Positive subjects":number_of_positive_subjects})
df.loc[:,"immunogenecity"] = (df.loc[:,"Positive subjects"] > 0).astype(int)

#%%
# Format MHCII for database
formatted_HLAs = []
for HLA in df["MHCs"]:
    if "/" in HLA:
        HLA = HLA[4:]
        first,second = HLA.split("/")
        HLA = first.replace("*","").replace(":","") + "-" + second .replace("*","").replace(":","")
    else:
        HLA = HLA[4:].replace("*","_").replace(":","")
    formatted_HLAs.append(HLA)
df.loc[:,"MHCs"] = formatted_HLAs

#%%
# Remove observations with MHCII not found in the database
HLA_database = read_HLA_database("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/MHCII/pseudosequence.2016.all.X.dat")    
unique_HLAs = set(df["MHCs"])
HLAs_in_db = [x for x in unique_HLAs if x in HLA_database.keys()]
df = df[df["MHCs"].isin(HLAs_in_db)]
#%%
# Remove negative observations with less then 4 tested subjects
df = df[~((df["immunogenecity"] == 0) & (df["Tested Subjects"] < 5))]
#%%
# Save DF for NetMHCIIpan predictions
df.to_csv("../data/df_before_binding_cores.csv")
# %%
