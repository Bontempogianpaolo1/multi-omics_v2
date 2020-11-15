from os import listdir

import pandas as pd
from sklearn.model_selection import train_test_split

from Classification.train_models import prepare_data
from Classification.train_models import test
from Classification.train_models import train_and_test

seed = 1200
# y
annotation_path = "annotation_global.csv"
files = ["Matrix_meth.csv",
         "Matrix_miRNA_deseq_correct.csv",
         "Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"]
train_path = "../Data/data/kidney/"
keys = ["meth", "miRNA", "mRNA"]

for filename,key in zip(files,keys):
    # read matrices
    X = pd.read_csv(train_path+filename)
    y = pd.read_csv(train_path+annotation_path)
    names = y["label"].astype('category').cat.categories
    # concat data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y["label"].astype('category').cat.codes,
                                                        shuffle=False, random_state=seed)
    X_train_transformed, y_train, X_test_transformed, y_test, pca = prepare_data(X_train, X_test, y_train, y_test,
                                                                                 n_components=21, names=names)
    models, modelnames = train_and_test(X_train_transformed, y_train, X_test_transformed, y_test, n_components=21,
                                        names=names, outputname="late-kidney-"+key)
    fromroot = "../Data/data"
    for directory in listdir(fromroot):
        if directory == "kidney":
            continue
        frompath = fromroot + "/" + directory + "/annotation_global.csv"
        y = pd.read_csv(frompath)
        frompath = fromroot + "/" + directory + "/" + filename
        X = pd.read_csv(frompath)
        X = X.fillna(0)
        X_test_transformed = pca.transform(X)
        test(X_test_transformed, y, models, modelnames, names, outputname="late-" + directory+"-"+key)
