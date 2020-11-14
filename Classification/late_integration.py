from os import listdir
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Classification.train_models import prepare_data
from Classification.train_models import test
from Classification.train_models import train_and_test

seed = 1200
# y
annotation_path = "../Data/data/kidney/annotation_global.csv"
meth_path = "../Data/data/kidney/Matrix_meth.csv"
mRNA_path = "../Data/data/kidney/Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/kidney/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path, mRNA_normalized_path, mRNA_path]
filenames = ["meth", "mRNA", "miRNA"]
y = pd.read_csv(annotation_path)
names = y["label"].astype('category').cat.categories
names2 = names.append(pd.Index(["Unknown"]))

for file, filename in zip(files, filenames):
    # read matrices
    X = pd.read_csv(file)
# concat data

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y["label"].astype('category').cat.codes,
                                                    shuffle=False, random_state=seed)
# reduce matrix
    X_train_transformed, y_train, X_test_transformed, y_test, pca = prepare_data(X_train, X_test, y_train, y_test,
                                                                             n_components=21)
    models, modelnames = train_and_test(X_train_transformed, y_train, X_test_transformed, y_test, n_components=21)

    fromroot = "./data"
    for directory in listdir(fromroot):
        data = []
        if directory == "kidney":
            continue
        for filename2 in listdir(fromroot + "/" + directory):
            frompath = fromroot + "/" + directory + "/" + filename
            if filename2.find(filename) != -1:
                X = pd.read_csv(frompath)
                data.append(X)
            elif filename2 == "annotation_global.csv":
                y = pd.read_csv(frompath)
                continue
            else:
                continue

            X_test_transformed = pca.transform(X)
            test(X_test_transformed, y, models, models, names)
