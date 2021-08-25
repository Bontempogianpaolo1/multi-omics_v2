import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import pandas as pd
import utils.custom_dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib as plt
from utils.Plot import plot_confusion_matrix
from rotation_forest import RotationForestClassifier

class RotationForest():
    def __init__(self):
        super(RotationForest, self).__init__()
        self.rf = RotationForestClassifier()

    def train_step(self, X, y_train, plot=False):
        y_train = y_train["label"].astype('category').cat.codes
        self.rf.fit(X,y_train)

    def test_forced(self, X, y_test):
        y_test = y_test["label"].astype('category').cat.codes
        probabilities=self.rf.predict_proba(X)
        return probabilities, y_test






if __name__ == "__main__":
    print("Mlp imported")
    num_iterations = 50
    num_features = 7
    seed = 1200
    annotation_path = "../../Data/data/kidney/preprocessed_annotation_global.csv"
    y = pd.read_csv(annotation_path)
    names = y["label"].astype('category').cat.categories
    #y = y.astype('category').cat.codes
    meth_path = "../../Data/data/kidney/preprocessed_Matrix_meth.csv"
    mRNA_path = "../../Data/data/kidney/preprocessed_Matrix_miRNA_deseq_correct.csv"
    mRNA_normalized_path = "../../Data/data/kidney/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
    files = [meth_path, mRNA_path, mRNA_normalized_path]
    filenames = ["meth", "miRNA", "mRNA"]
    modelname = "mlp"
    for file, filename in zip(files, filenames):
        with open('../Data/outputs/' + filename + '-bnn-output.txt', 'w') as f:
            X = pd.read_csv(file, index_col=False, header=None)
            if filename == "mrna":
                X = pd.DataFrame(X[X.std().sort_values(ascending=False).head(1200).index].values.tolist())
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y["label"])
            pca = PCA(n_components=num_features)
            X_train_transformed = pca.fit_transform(X_train)
            X_test_transformed = pca.transform(X_test)
            clf = MLP(num_features, 20, 5)
            clf.train_step(X_train_transformed, y_train)
            probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)
            print(filename)

            pd.DataFrame(probabilities).to_csv("../Data/outputs/pred-mlp-" + filename + ".csv")
            pd.DataFrame(true_labels).to_csv("../Data/outputs/true-labels.csv")

            X2 = pd.read_csv("../Data/data/anomalies_preprocessed_Matrix_" + filename + ".csv", index_col=False,
                             header=None)
            y2 = pd.read_csv("../Data/data/stomach/anomalies_preprocessed_annotation_global.csv")["label"]
            if filename == "mrna":
                X2 = pd.DataFrame(X2[X2.std().sort_values(ascending=False).head(1200).index].values.tolist())
            X_transformed = pca.transform(X2)
            print('Prediction when network is forced to predict')

            probabilities, true_labels = clf.test_forced(X_transformed, y2.astype('category').cat.codes)
            y_pred=  np.argmax(probabilities, axis=1)
            y_pred[np.max(probabilities,axis=1)<0.6]=5
            cnf_matrix = confusion_matrix(true_labels[1:], np.argmax(probabilities, axis=1))
            # plt.figure(figsize=(10, 10))
            # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
            print()
            np.set_printoptions(precision=2)
            # PlotDir non-normalized confusion matrix
            plt.figure.Figure(figsize=(10, 10))

            plot_confusion_matrix(cnf_matrix,
                                  title=modelname + "-anomalies-" + filename,
                                  classes=names)
    #plot_mlp_results()
