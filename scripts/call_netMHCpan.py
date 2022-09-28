import os
import sys
from tqdm import tqdm
import time
## Used in computerome to use the netMHCpan-4.1 tool.
#$NETMHCpanII -f $path/$protein.fasta -a $alleles -filter -rankF $wb_threshold -BA -context -termAcon -length $number -rankS $sb_threshold -rankW $wb_threshold > $results/pred.$protein.$number.$sb_threshold.$wb_threshold

## Basically reads the peptide file and calls the netMHCpan-4.1 tool for the specific allele and outputs in given file

path_to_fasta_files = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/fastas"
path_to_results = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/newMHCIIpan_results"
path_to_netMHCIIpan = "/Users/christianpederjacobsen/tools/netMHCIIpan-4.0/netMHCIIpan"
alleles = "DRB1_1454,DRB1_1601,DRB1_1401,DRB1_1504,DRB1_0101,DRB1_1201,DRB1_0801,DRB1_1101,DRB1_1301,DRB1_0301,DRB1_1104,DRB1_1501,DRB1_0701,DRB1_1302,DRB1_0404,DRB1_0405,DRB1_0901,DRB1_0401,DRB1_1503,DRB1_1602,DRB1_0804,DRB1_1001,DRB1_0302,DRB1_1103,DRB1_0407,DRB1_1102,DRB4_0101,DRB3_0202,DRB3_0101,DRB3_0301,DRB3_0201,DRB5_0101,DRB5_0202,DRB5_0102"
counter = 0 
batch_counter = 0
# print("First batch")
for file in tqdm(os.listdir(path_to_fasta_files),total=len(os.listdir(path_to_fasta_files))):
    protein = file.replace(".fasta","")
    outfile_suffix = f"pred.{protein}.1.5"
    path_to_fasta_file = os.path.join(path_to_fasta_files,file)
    path_to_netMHC_results = os.path.join(path_to_results,outfile_suffix)
    # if counter % 10 == 0 and counter != 0:
    #     time.sleep(60)
    #     batch_counter += 1
    #     print(f"New batch - {batch_counter}")
    os.system(f"{path_to_netMHCIIpan} -f {path_to_fasta_file} -a \"{alleles}\" -filter -BA -context -length 15 -rankS 1 -rankW 5 > {path_to_netMHC_results}")
    # counter += 1