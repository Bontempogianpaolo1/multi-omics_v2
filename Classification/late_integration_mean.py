import ast

import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from utils.Plot import plot_confusion_matrix
outputpath = "../Data/outputs/"
annotation_path = "../Data/data/kidney/preprocessed_annotation_global.csv"
names = pd.read_csv(annotation_path)["label"].astype('category').cat.categories
modelnames = ["bnn", "mlptree", "mlp"]
filenames = ["miRNA", "meth", "mRNA"]
datasets = ["kidney", "stomach", "lung"]
conservative = [True, False]
# testset
for dataset in datasets:
    path = "../Data/outputs/pred-testset-"
    if dataset == "kidney":
        for modelname in modelnames:
            data = []
            for filename in filenames:
                X = pd.read_csv(path + modelname + "-" + filename + ".csv")
                probabities = torch.tensor([]).new_empty((X.shape[0], 5))
                for i, row in enumerate(X["probabilities"]):
                    probabities[i] = torch.tensor(ast.literal_eval(row))
                data.append(probabities)

            matrix = torch.stack([data[0], data[1], data[2]], 2)
            y_pred = torch.argmax(matrix.sum(2), dim=1)
            y_true = X['y_true']
            for is_conservative in conservative:
                if is_conservative:
                    pathto = outputpath+"/with-unknown-"+dataset+"-"+modelname
                    constraint = torch.div(matrix.sum(2), matrix.sum(2).sum(1).view(y_pred.shape[0], 1))
                    max_prob = torch.max(matrix.sum(2), dim=1)
                    max_constraint = torch.max(constraint, dim=1)
                    y_pred[(max_prob.values / 3 < 0.9) | (max_constraint.values < 0.25)] = 5
                else:
                    pathto = outputpath+"/without-unknown-"+dataset+"-"+modelname

                cnf_matrix = confusion_matrix(y_true, y_pred)
                print("plot")
                np.set_printoptions(precision=2)
                # PlotDir non-normalized confusion matrix
                plt.figure.Figure(figsize=(10, 10))
                plot_confusion_matrix(cnf_matrix,title=pathto+ "-confusionmatrix", classes=names)
                with open(pathto + "-comparison-new.txt", 'w') as f:
                    print("total predicted as unknown " + str((y_pred.numpy() == 5).sum()), file=f)
                    y_true2 = y_true[y_pred.numpy() != 5]
                    y_pred2 = y_pred[y_pred.numpy() != 5]
                    print(classification_report(y_true2, y_pred2, ), file=f)