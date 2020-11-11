import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples


def getdistances():
    seed = 1200
    annotation_path = "../Data/data/kidney/preprocessed_annotation_global.csv"
    y = pd.read_csv(annotation_path)
    # y = pd.read_csv(annotation_path)["label"]
    # names = y.astype('category').cat.categories
    y = y['label'].astype('category').cat.codes
    modelname = " mlp "
    meth_path = "../Data/data/kidney/preprocessed_Matrix_meth.csv"
    mRNA_path = "../Data/data/kidney/preprocessed_Matrix_miRNA_deseq_correct.csv"
    mRNA_normalized_path = "../Data/data/kidney/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
    files = [meth_path,  mRNA_normalized_path,mRNA_path]
    filenames = ["meth","mRNA", "miRNA"]
    parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
                  'max_iter': [200, 400, 600]}
    true_labels = []
    omics=[]
    for file, filename in zip(files, filenames):
        outputname = modelname + filename
        X = pd.read_csv(file, index_col=False, header=None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed,
                                                            stratify=y)
        n_components = 7
        pca = PCA(n_components=n_components)

        X_train_transformed = pca.fit_transform(X_train)
        sample_silhouette_values = silhouette_samples(X_train_transformed, y_train)
        #X_train_transformed = X_train
        weigths = []
        for n_cluster in range(5):
            cluster_avg= sample_silhouette_values[y_train==n_cluster].mean()
            weigths.append(cluster_avg)

        omics.append(weigths/max(weigths))
    return np.array(omics).T
