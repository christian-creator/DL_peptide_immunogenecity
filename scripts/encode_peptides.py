import pandas as pd
import numpy as np
import sys
import os

def encode_peptide_AAindex(peptide):
    """Encode peptides using principal components derived from the AAindex database describing ~90% variance of all features derived from the AAindex database

    Args:
        peptide (str): peptide to encode

    Returns:
        np.array: encoded peptides
    """
    PCA_repr = pd.read_csv("data/PCA_repr_aa.csv",index_col=0)
    encoded_peptide = []
    for aa in list(peptide):
        if aa == "-":
            n_pcs = len(PCA_repr.columns)
            encoded_peptide.append([0 for _ in range(n_pcs)])
        else:
            encoded_peptide.append(list(PCA_repr.loc[aa]))
    return np.array(encoded_peptide)
    ...


# TODO: Use other encoding schemes like sparse onehot encoding or BLOSSUM
def encode_peptide_onehot(peptide):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F','P', 'S', 'T', 'W', 'Y', 'V']
    ...

def encode_seq_vec(peptide):
    ...

# TODO: Encode using SeqVec: https://github.com/Rostlab/SeqVec
if __name__ == "__main__":
    os.system("ls ../")
    encode_peptide_AAindex("AAVKAGAAL-")