import os 
import sys

def read_fasta(fasta_file):
    name, seq = None, []
    for line in fasta_file:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))

path_to_fasta_file = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/all_human_theraputics.fa"
outfile = open("./Humab_theraputics_data/9_mer_database_humab.txt",'w+')

with open(path_to_fasta_file,'r') as infile:
    for name, seq in read_fasta(infile):
        print(name, seq)
        # initializing K
        K = 9
        # extracting window using slicing
        res = [seq[idx:idx + K] for idx in range(len(seq) - K + 1)]
        print("\n".join(res),file=outfile)