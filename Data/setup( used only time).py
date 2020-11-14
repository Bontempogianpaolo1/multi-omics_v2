from os import listdir

import pandas as pd

root = "./original"
for directory in listdir(root):
    for filename in listdir(root + "/" + directory):
        path = root + "/" + directory + "/" + filename
        prova = root + "/" + directory + "/transposed-" + filename
        if filename == "annotation_global.csv":
            continue
        if filename == "Matrix_meth.csv":
            file = pd.read_csv(path, sep='\t').transpose()
        elif filename == "Matrix_miRNA_deseq_correct.csv":
            file = pd.read_csv(path, sep=',').transpose()
        elif filename == "Matrix_mRNA_deseq_normalized_prot_coding_correct.csv":
            file = pd.read_csv(path, sep=',').transpose()
        elif filename == "Matrix_meth.csv":
            file = pd.read_csv(path, sep=',').transpose()
        else:
            continue
        print("read")
        print(prova)
        print(file.shape)
        file.to_csv(prova,header=False)
        print("written")
