import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import sys
import seaborn as sns
from matplotlib import cm
import scipy
from sklearn.model_selection import KFold
from scipy.stats import betabinom
import copy
import itertools


def calculate_immunogenecity_score(tested_subjects,positive_subjects):
    N = tested_subjects
    m = positive_subjects
    a = 1 + m
    b = 1 + N - m
    score = betabinom.mean(1,a, b)
    return score

## Read epitope data
# path_to_epitope_file = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/eptope_data_MHCII.csv"
path_to_epitope_file = "data/MHCI/epitope_data_MHCI.csv"
epitope_raw_df = pd.read_csv(path_to_epitope_file,skiprows=1)
epitope_raw_df = epitope_raw_df[['Epitope ID',"Description","Antigen Name"]]
## Read experiment data
# path_to_experiment_file = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/experiment_data_MHCII.csv"
path_to_experiment_file = "data/MHCI/experiment_data_MHCI.csv"
experiment_raw_df = pd.read_csv(path_to_experiment_file,skiprows=1)
experiment_raw_df = experiment_raw_df[['Epitope ID',"Cell Type","Allele Name","Antigen Description","Qualitative Measure","Assay Group","Number of Subjects Tested","Number of Subjects Responded"]].dropna()
experiment_raw_df.sort_values(by='Epitope ID', ascending=True)

# Experiments used 
# most_common_experiments = ['IFNg release', 'qualitative binding', 'cytotoxicity', 'activation']
used_experiments = ['IFNg release']
# used_experiments = ['IFNg release', 'qualitative binding', 'cytotoxicity', 'activation']

filter_lengths = [9,10]
epitope_raw_df["epitope_length"] = epitope_raw_df["Description"].apply(lambda x: len(x.strip()))
epitope_df = epitope_raw_df[epitope_raw_df.epitope_length.isin(filter_lengths)]

epitope_IDS = epitope_df["Epitope ID"]
final_dataset = dict()

for ID in epitope_IDS:
    if len(experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]) == 0:
        continue

    number_of_assay_groups_for_id = len(set(experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]["Assay Group"]))
    number_of_HLA_alleles = len(set(experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]["Allele Name"]))

    peptide = list(epitope_df[epitope_df["Epitope ID"] == ID]["Description"])[0]

    experiments_with_peptide = experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]
    

    for i,experiment in experiments_with_peptide.iterrows():
        if experiment["Antigen Description"] == peptide:
            
            unique_identifier = peptide + "_" + experiment["Allele Name"]

            assay = experiment["Assay Group"]
            if assay not in used_experiments:
                continue

            if str(ID) == "75636":
                print("Unique identifier", unique_identifier)
                print(experiment)
                print()


            tested_subjects = experiment["Number of Subjects Tested"]
            positive_subjects = experiment["Number of Subjects Responded"]

            if unique_identifier not in final_dataset.keys():
                final_dataset[unique_identifier] = dict()

            if assay not in final_dataset[unique_identifier].keys():
                final_dataset[unique_identifier]["tested subjects"] = [tested_subjects]
                final_dataset[unique_identifier]["positive subjects"] = [positive_subjects]
            else:
                final_dataset[unique_identifier]["tested subjects"].append(tested_subjects)
                final_dataset[unique_identifier]["positive subjects"].append(positive_subjects)
    if str(ID) == "75636":
        print(final_dataset[unique_identifier])
        sys.exit(1)



dataset = []
for peptide_HLA in final_dataset.keys():
    peptide,HLA_allele = peptide_HLA.split("_")
    tested_subjects = np.sum(final_dataset[peptide_HLA]["tested subjects"])
    positive_subjects = np.sum(final_dataset[peptide_HLA]["positive subjects"])
    immunogenicity = calculate_immunogenecity_score(tested_subjects,positive_subjects)    
    dataset.append([peptide,HLA_allele,immunogenicity,tested_subjects,positive_subjects])


columns = ["peptide","HLA_allele","immunogenicity","tested_subjects","positive_subjects"]
dataset_df = pd.DataFrame.from_records(dataset,columns=columns)
alleles = ["*" in x for x in dataset_df["HLA_allele"]]
dataset_df = dataset_df[alleles]


positive_df = dataset_df[dataset_df["positive_subjects"] > 0]
negative_df = dataset_df[dataset_df["positive_subjects"] == 0]
negative_df = negative_df[negative_df["tested_subjects"] > 4]
total_df = pd.concat([positive_df, negative_df])


#positive_df.to_csv("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/positive_df_tmp.csv")
#negative_df.to_csv("/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/data/negative_df_tmp.csv")

total_df["Response"] = ['Positive' if x > 0 else 'Negative' for x in total_df["positive_subjects"]]

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

hla = pd.read_csv('data/formatted_MHC_pseudo.dat', sep=" ")
hla_dic = hla_df_to_dic(hla)

removed_idx = total_df[total_df.apply(lambda x: False if (x.HLA_allele) in hla_dic.keys() else True,axis=1)].index
print("These rows were removed due to HLA missing from the MHC_psuedo.dat: {}".format(removed_idx))
total_df = total_df[total_df.apply(lambda x: True if (x.HLA_allele) in hla_dic.keys() else False,axis=1)]
# total_df.to_csv("../tmp/filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay.csv")
