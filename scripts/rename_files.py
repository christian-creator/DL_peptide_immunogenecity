import os
import sys

path_to_folder = "/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/peptide_immunogenicity/MHCII/Humab_theraputics_data/newMHCIIpan_results"

for file in os.listdir(path_to_folder):
    path_to_file = os.path.join(path_to_folder,file)
    file_split = file.split(".")
    new_file_name = ".".join(file_split[:2] + ["15"] + file_split[2:])
    path_to_new_file = os.path.join(path_to_folder,new_file_name)
    os.system(f"mv {path_to_file} {path_to_new_file}")
    # print(f"mv {path_to_file} {path_to_new_file}")