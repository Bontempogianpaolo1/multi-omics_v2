import matplotlib as plt
import numpy as np
import pandas as pd
import ast
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.Plot import plot_confusion_matrix
annotation_path = "../Data/data/kidney/preprocessed_annotation_global.csv"
names = pd.read_csv(annotation_path)["label"].astype('category').cat.categories
modelnames = [ "bnn","mlptree", "mlp"]
filenames = ["miRNA", "meth", "mRNA"]
#testset
path="../Data/outputs/pred-testset-"
counters=torch.Tensor().new_zeros((228))
for modelname in modelnames:
    data=[]
    for filename in filenames:
        X = pd.read_csv(path+ modelname + "-" + filename + ".csv")
        probabities= torch.tensor([]).new_empty((X.shape[0],5))
        for i,row in enumerate(X["probabilities"]):
            probabities[i]=torch.tensor(ast.literal_eval(row))
        data.append(probabities)

    matrix=torch.stack([data[0],data[1],data[2]],2)
    y_pred=torch.argmax(matrix.sum(2), dim=1)
    constraint= torch.div(matrix.sum(2),matrix.sum(2).sum(1).view(y_pred.shape[0],1))
    max_prob=torch.max(matrix.sum(2),dim=1)
    max_constraint = torch.max(constraint, dim=1)
    #y_pred[(max_prob.values/3<0.9) |(max_constraint.values<0.25)]=5
    y_true = X['y_true']
    #counters=y_pred.new_zeros(y_pred.shape)
    counters[y_pred==5]=counters[y_pred==5]+1

    #y_true=y_true[max_prob.values.numpy()/3<0.9]

    print("plot")
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 10))
    # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
    print()
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))

    plot_confusion_matrix(cnf_matrix,
                          title="with-unknown-testset-" + modelname + "-comparison-new",classes=names)
    with open("../Data/outputs/with-unknown-testset-" + modelname + "-comparison-new.txt", 'w') as f:

        print(classification_report(y_true, y_pred, ), file=f)
    with open("../Data/outputs/without-unknown-testset-" + modelname + "-comparison-new.txt", 'w') as f:
        print("total predicted as unknown "+ str((y_pred.numpy()==5).sum()),file=f)
        y_true2 = y_true[y_pred.numpy() != 5]
        y_pred2 = y_pred[y_pred.numpy() != 5]
        print(classification_report(y_true2, y_pred2, ), file=f)

path="../Data/outputs/pred-stomaco-"
with open("../Data/outputs/counters-testset.txt", 'w') as f:
    print("total unknown" + str((counters!=0).sum()),file=f)
    print("unknown for one model" + str((counters == 1).sum()), file=f)
    print("unknown for two models" + str((counters == 2).sum()), file=f)
    print("unknown for three models" + str((counters == 3).sum()), file=f)
counters=torch.Tensor().new_zeros((37))
for modelname in modelnames:
    data=[]
    for filename in filenames:
        X = pd.read_csv(path+ modelname + "-" + filename + ".csv")
        probabities = torch.tensor([]).new_empty((X.shape[0], 5))
        for i, row in enumerate(X["probabilities"]):
            probabities[i] = torch.tensor(ast.literal_eval(row))
        data.append(probabities)
    matrix = torch.stack([data[0], data[1], data[2]], 2)
    y_pred = torch.argmax(matrix.sum(2), dim=1)
    max_prob = torch.max(matrix.sum(2), dim=1)
    y_pred[max_prob.values / 3 < 0.9] = 5
    counters[y_pred == 5] = counters[y_pred == 5] + 1
    y_true = X['y_true']

    y_true[y_true != 5] = 0
    y_true[y_true == 5] = 1
    y_pred[y_pred != 5] = 0
    y_pred[y_pred == 5] = 1


    cnf_matrix = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 10))
    # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
    print()
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))

    plot_confusion_matrix(cnf_matrix,
                          title="with-unknown-stomaco-" + modelname + "-comparison-new",classes=["predicted","Unknown"])
with open("../Data/outputs/counters-stomaco.txt", 'w') as f:
    print("total unknown" + str((counters!=0).sum()),file=f)
    print("unknown for one model" + str((counters == 1).sum()), file=f)
    print("unknown for two models" + str((counters == 2).sum()), file=f)
    print("unknown for three models" + str((counters == 3).sum()), file=f)