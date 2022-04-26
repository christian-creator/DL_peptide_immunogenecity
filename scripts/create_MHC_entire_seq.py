# %% 
import os
import outcome
import pandas as pd
import sys

def create_fasta_of_alleles_in_dataframe(HLA_alleles,alleles_in_dataframe,file_name):
    outfile = "../tmp/MHC_full_in_big_data.fasta"
    outfile = file_name
    with open(outfile,"w+") as outfile:
        for MHCI_allele in alleles_in_dataframe:
            print(">",MHCI_allele,sep="",file=outfile)
            print(HLA_alleles[MHCI_allele],file=outfile)

def read_aligned_file(path_to_aligned_file):
    sequence_data = dict()
    with open(path_to_aligned_file,"r") as f:
        for line in f:
            if line[0] == ">":
                header = line[1:].strip()
                sequence_data[header] = ""
            else:
                sequence_data[header] += line.strip()
    return sequence_data

def create_MHC_file(sequence_data,outfile_name):
    with open(outfile_name,"w+") as outfile:
        print("HLA","pseudo",file=outfile)
        for HLA in sequence_data.keys():
            hla_formatted = HLA.replace(":","")
            print(hla_formatted,sequence_data[HLA],file=outfile)




# %%
file_prot_a_filename = "../tmp/A_prot.fasta"
file_prot_b_filename = "../tmp/B_prot.fasta"
file_prot_c_filename = "../tmp/C_prot.fasta"

HLA_alleles = dict()
file_names = [file_prot_a_filename,file_prot_b_filename,file_prot_c_filename] 
read_flag = False
for filename in file_names:
    with open(filename,"r") as f:
        for line in f:
            if line[0] == ">":
                MHC_allele_long_code = line.split()[1]
                first_two_digits  = MHC_allele_long_code.split(":")[0][2:]
                second_two_digits = MHC_allele_long_code.split(":")[1]
                letter = MHC_allele_long_code[0]
                # print(MHC_allele_long_code,letter,first_two_digits, second_two_digits)
                allele_name = f"HLA-{letter}*{first_two_digits}:{second_two_digits}"
                if allele_name not in HLA_alleles.keys():
                    HLA_alleles[allele_name] = ""
                    read_flag = True
                else:
                    read_flag = False
                    continue
                    
            else:
                if read_flag:
                    HLA_alleles[allele_name] += line.strip()

# %%
all_data = pd.read_csv("../data/filtered_data_IEDB_4_tested_len_9_10_full_HLA_Multi_assay_w_binding_w_parts.csv")
alleles_in_dataframe = list(set([x for x in all_data["HLA_allele"] if x in HLA_alleles.keys()]))

# %%
sequenced_data = read_aligned_file("../tmp/MHC_full_aligned.dat")
create_MHC_file(sequenced_data,"../data/MHC_full.dat")
# %%

# %%
