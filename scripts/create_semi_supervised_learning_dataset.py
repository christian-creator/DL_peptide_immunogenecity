# %%
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
# %%
def plot_barplot(feature,xlabel):
    unique_entries, counts = np.unique(feature,return_counts=True)
    count_dict = {hla:count for hla,count in zip(unique_entries, counts) }

    sorted_HLAs = sorted(count_dict.keys(),key=lambda x: count_dict[x], reverse=True) 
    sorted_counts = [count_dict[x] for x in sorted_HLAs]

    fig = plt.figure(figsize=(20,6))
    plt.bar(np.arange(len(unique_entries)),sorted_counts)
    plt.xticks(np.arange(len(unique_entries)),sorted_HLAs,rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.show()

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

def calculate_identity(query,database):
    if len(query) == len(database):
        return np.sum([1 for i in range(len(query)) if query[i] == database[i]])/len(query)
    
    elif len(query) > len(database):
        max_score = 0
        diff_len = len(query) - len(database)
        for i in range(diff_len + 1):
            query_shortened = query[i:i+len(database)]
            score = np.sum([1 for i in range(len(query_shortened)) if query_shortened[i] == database[i]])/len(query_shortened)
            if score > max_score:
                max_score = score

        return max_score
    
    elif len(query) < len(database):
        max_score = 0
        diff_len = len(database) - len(query)
        for i in range(diff_len + 1):
            database_shortened = database[i:i+len(query)]
            score = np.sum([1 for i in range(len(query)) if query[i] == database_shortened[i]])/len(database_shortened)
            if score > max_score:
                max_score = score

        return max_score
# %%
path_to_epitope_file = "../data/MHCI_binding/epitope_table_export_1647521521.csv"
epitope_raw_df =  pd.read_csv(path_to_epitope_file,skiprows=1,low_memory=False)
epitope_raw_df = epitope_raw_df[['Epitope ID',"Description","Antigen Name"]]
epitope_raw_df.head()
# %%
path_to_experiment_file = "../data/MHCI_binding/mhc_ligand_table_export_1647533328.csv"
experiment_raw_df = pd.read_csv(path_to_experiment_file,skiprows=1,low_memory=False)
experiment_raw_df = experiment_raw_df[['Epitope ID',"Allele Name","Qualitative Measure","Assay Group","Description"]]
experiment_raw_df.sort_values(by='Epitope ID', ascending=True)
# plot_barplot(experiment_raw_df["Assay Group"],"Assay group")
# plot_barplot(experiment_raw_df["Qualitative Measure"],"Qualitative Measure")
# experiment_raw_df.isnull().sum().sort_values(ascending = False)

# %%
used_experiments = ['dissociation constant KD (~IC50)', 'half maximal inhibitory concentration (IC50)','ligand presentation','qualitative binding','dissociation constant KD (~EC50)']
filter_lengths = [9,10]
# %%
epitope_raw_df["epitope_length"] = epitope_raw_df["Description"].apply(lambda x: len(x.strip()))
epitope_df = epitope_raw_df[epitope_raw_df.epitope_length.isin(filter_lengths)]
epitope_IDS = epitope_df["Epitope ID"]
final_dataset = dict()
# %%
print("## Creating dataframe from raw data ")
for ID in epitope_IDS:
    if len(experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]) == 0:
        continue
    number_of_assay_groups_for_id = len(set(experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]["Assay Group"]))
    number_of_HLA_alleles = len(set(experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]["Allele Name"]))

    peptide = list(epitope_df[epitope_df["Epitope ID"] == ID]["Description"])[0]
    experiments_with_peptide = experiment_raw_df[experiment_raw_df["Epitope ID"] == ID]
    
    
    for i,experiment in experiments_with_peptide.iterrows():
        if experiment["Description"] == peptide:
            unique_identifier = peptide + "_" + experiment["Allele Name"]
            assay = experiment["Assay Group"]
            qualitative_measure = experiment['Qualitative Measure']

            if assay not in used_experiments or qualitative_measure not in ["Positive","Positive-High"]:
                continue

            if unique_identifier not in final_dataset.keys():
                final_dataset[unique_identifier] = dict()

            if assay not in final_dataset[unique_identifier].keys():
                final_dataset[unique_identifier]["Qualitative Measure"] = qualitative_measure

# %%
dataset = []
for peptide_HLA in final_dataset.keys():
    peptide,HLA_allele = peptide_HLA.split("_")
    qualitive_measure = final_dataset[peptide_HLA]['Qualitative Measure']
    dataset.append([peptide,HLA_allele,qualitive_measure])

columns = ["peptide","HLA_allele","Qualitative Measure"]
unsupervised_df = pd.DataFrame.from_records(dataset,columns=columns)
alleles = ["*" in x for x in unsupervised_df["HLA_allele"]]
unsupervised_df = unsupervised_df[alleles]
# %%
hla = pd.read_csv('../data/formatted_MHC_pseudo.dat', sep=" ")
hla_dic = hla_df_to_dic(hla)
removed_idx = unsupervised_df[unsupervised_df.apply(lambda x: False if (x.HLA_allele) in hla_dic.keys() else True,axis=1)].index
print("These rows were removed due to HLA missing from the MHC_psuedo.dat: {}".format(removed_idx))
unsupervised_df = unsupervised_df[unsupervised_df.apply(lambda x: True if (x.HLA_allele) in hla_dic.keys() else False,axis=1)]

# %%
path_to_dataset = "../data/filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay_w_binding.csv"
multi_assay_df = pd.read_csv(path_to_dataset)

# %%
print("## Checking for overlap to immunogenecity data")
black_list_indexes = []
for i,unsupervised_row in tqdm(unsupervised_df.iterrows()):
    uns_peptide,uns_HLA = unsupervised_row['peptide'], unsupervised_row['HLA_allele']
    for j,imm_row in multi_assay_df.iterrows():
        imm_peptide, imm_hla = imm_row['peptide'], imm_row['HLA_allele']
        identitiy_score = calculate_identity(uns_peptide,imm_peptide)
        if identitiy_score == 1.0 and uns_HLA == imm_hla:
            black_list_indexes.append(i)
        

# %%
unsupervised_df = unsupervised_df.drop(unsupervised_df.index[black_list_indexes])
unsupervised_df.to_csv("../data/semi_supervised_data_w_binding_no_overlap.csv")
# %%
