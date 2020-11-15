from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split
from Classification.train_models import prepare_data
from Classification.train_models import test
from Classification.train_models import train_and_test

seed = 1200
# y
annotation_path = "../Data/data/kidney/annotation_global.csv"
y = pd.read_csv(annotation_path)
names = y["label"].astype('category').cat.categories
# filenames
files = ["../Data/data/kidney/Matrix_meth.csv",
         "../Data/data/kidney/Matrix_miRNA_deseq_correct.csv",
         "../Data/data/kidney/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"]
filenames = ["meth", "miRNA", "mRNA"]
data = []
# read labels

for file, filename in zip(files, filenames):
    # read matrices
    X = pd.read_csv(file)
    data.append(X)
# concat data
X = pd.concat([data[0], data[1], data[2]], axis=1)
X = pd.DataFrame(X.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y["label"].astype('category').cat.codes,
                                                    shuffle=False, random_state=seed)
# reduce matrix
X_train_transformed, y_train, X_test_transformed, y_test, pca = prepare_data(X_train, X_test, y_train, y_test,
                                                                             n_components=21, names=names)
models, modelnames = train_and_test(X_train_transformed, y_train, X_test_transformed, y_test, n_components=21,
                                    names=names,outputname="early-kidney")
fromroot = "../Data/data"
for directory in listdir(fromroot):
    data = []
    if directory == "kidney":
        continue
    for filename in listdir(fromroot + "/" + directory):
        frompath = fromroot + "/" + directory + "/" + filename
        if (filename == "Matrix_meth.csv") or (filename == "Matrix_miRNA_deseq_correct.csv") or (
                filename == "Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"):
            X = pd.read_csv(frompath)
            data.append(X)
        elif filename == "annotation_global.csv":
            y = pd.read_csv(frompath)
        else:
            continue
    X = pd.concat([data[0], data[1], data[2]], axis=1)
    X=X.fillna(0)
    X_test_transformed = pca.transform(X)
    test(X_test_transformed, y, models, modelnames, names, outputname="early"+directory)
