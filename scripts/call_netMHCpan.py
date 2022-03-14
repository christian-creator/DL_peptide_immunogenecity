import os
## Used in computerome to use the netMHCpan-4.1 tool.

## Basically reads the peptide file and calls the netMHCpan-4.1 tool for the specific allele and outputs in given file

path_to_peptide_files = "/home/projects/vaccine/people/chrjac/DL_project/peptides_file"
path_to_results = "/home/projects/vaccine/people/chrjac/DL_project/netMHCpan_results"
for file in os.listdir(path_to_peptide_files):
    allele = file.split("_p")[0]
    allele = allele.replace("*","")
    path_to_peptide_file = os.path.join(path_to_peptide_files,file)
    path_to_netMHC_results = os.path.join(path_to_results,file)
    os.system(f"netMHCpan-4.1 -p -f {path_to_peptide_file} -a \"{allele}\" > {path_to_netMHC_results}")