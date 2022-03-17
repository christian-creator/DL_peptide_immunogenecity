# %% 
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# %% 
def read_netMHCpan_results(path_to_results_files):
    """Reads the netMHCpan results generated from the peptides of bound to each MHCI allele in the filtered dataset

    Args:
        path_to_results_files (str): path_to_results

    Returns:
        dict: binding score of each MHCI-peptide pair
    """
    results = dict()
    for file in os.listdir(path_to_results_files):
        path_to_file = os.path.join(path_to_results_files,file)
        with open(path_to_file,"r") as f:
            for line in f:
                line = line.split()
                if len(line) > 1:
                    if line[0] == "1":
                        HLA = line[1]
                        peptide = line[2]
                        score_EL = line[11]
                        rank_EL = line[12]
                        score = float(score_EL)
                        if peptide not in results.keys():
                            results[peptide] = {HLA:score}
                        else:
                            results[peptide][HLA] = score

    return results
# %% 
# Reading the filtered dataset
data = pd.read_csv("../data/filtered_data_IEDB_4_tested_len_9_10_full_HLA.csv")

# Reading the binding results from netMHCpan output
path_to_netMHCpan_results = "../data/netMHCpan_results"
binding_results = read_netMHCpan_results(path_to_netMHCpan_results)

# Making a binary classication feature based on a threshold 0.5 and the continous immunogenecity scores. 
data["Binary clasification"] =  [1 if x > 0.5 else 0 for x in data["immunogenicity"]]

# %%
# Adding the results to the DataFram
all_binding_score = []
for i,row in data.iterrows():
    peptide = row["peptide"]
    HLA = row["HLA_allele"]
    binding_score = binding_results[peptide][HLA]
    all_binding_score.append(binding_score)
data["Binding Scores"] = all_binding_score

# %%
# Making scatter plot of the binding scores against the immunogenecity
fig = plt.figure(figsize=(10,6))
sns.scatterplot(data=data,x="Binding Scores",y="immunogenicity")
plt.show()
# %%
# Calculating the ROC curve and the AUC score
fpr, tpr, threshold = metrics.roc_curve(data["Binary clasification"],data["Binding Scores"])
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("## THUS WE GET AN AUC OF 0.6 WHEN ONLY USING THE BINDING SCORE TO PREDICT IMMUNOGENICITY")
# %%
