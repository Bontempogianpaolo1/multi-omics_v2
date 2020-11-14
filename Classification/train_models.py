import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import Classification.models as models
import Classification.models.Mlp as Mlp
import Classification.models.Mlptree as Mlptree
import Classification.models.bnn as bnn
from utils.Plot import PlotInstograms
from utils.Plot import plot_confusion_matrix


def prepare_data(x_train, x_test, y_train, y_test, n_components, names):
    # increment samples -----------------------------------------------------
    # names = y["label"].astype('category').cat.categories
    sme = SMOTE(random_state=42)
    x_res, y_res = sme.fit_sample(x_train, y_train["label"].to_numpy())
    x_new = x_res[681:]
    y_res = y_res[681:]
    for label in names:
        x_cur = x_new[y_res == label]
        tot = (y_res == label).sum()
        if tot != 0:
            df_to_add = pd.DataFrame({
                'official_name': np.ndarray((y_res == label).sum()).tolist(),
                'project_id': np.ndarray((y_res == label).sum()).tolist(),
                'case_id': np.ndarray((y_res == label).sum()).tolist(),
                'label': y_res[y_res == label].tolist()
            })
            df_to_add['is_tumor'] = df_to_add['label'].str.split('-').str[0]
            y_train = y_train.append(df_to_add)
            x_train = x_train.append(x_cur)

    # -------------------------------------------------------------------------------
    pca = PCA(n_components=n_components)
    x_train_transformed = pca.fit_transform(x_train)
    x_test_transformed = pca.transform(x_test)
    return x_train_transformed, y_train, x_test_transformed, y_test,pca


def train_and_test(x_train_transformed, y_train, x_test_transformed, y_test, n_components, names):
    models = []
    modelnames = ["mlptree", "mlp", "bnn"]
    for model, modelname in modelnames:
        scores = np.empty([])
        # train and test bnn
        if modelname == "bnn":
            clf = models.bnn.BNN(n_components, 20, 5)
            clf.train_step(x_train_transformed, y_train)
            tot, correct_predictions, predicted_for_images, new_prediction, probabilities = clf.test_batch(
                torch.from_numpy(x_test_transformed).float(), y_test, names, plot=False)
            y_pred = new_prediction
            maxprob = np.max(probabilities, axis=1)

        elif modelname == "mlp":
            # train and test mlp
            clf = models.Mlp.MLP(n_components, 20, 5)
            clf.train_step(x_train_transformed, y_train)
            probabilities, true_labels = clf.test_forced(x_test_transformed, y_test)
            y_pred = np.argmax(probabilities, axis=1)
            maxprob = np.max(probabilities, axis=1)
            y_pred[maxprob < 0.9] = 5

        elif modelname == "mlptree":
            # train and test mlptree
            clf = models.Mlptree.MlpTree(n_components, 20, 5)
            clf.train_step(x_train_transformed, y_train)
            maxprob, y_pred, true_labels, probabilities = clf.test_forced(x_test_transformed, y_test)
            y_pred[maxprob < 0.9] = 5
        else:
            return

        # save results
        df = pd.DataFrame({
            'official_name': y_test['official_name'].tolist(),
            'max_probability': maxprob.tolist(),
            'probabilities': np.array(probabilities).tolist(),
            'y_pred': y_pred.tolist(),
            'y_true': y_test["label"].astype('category').cat.codes.tolist()
        })
        df.to_csv("../Data/outputs/pred-testset-" + modelname + "-" + "-onlyonedataset.csv")
        PlotInstograms(df, "istogramma testset unico dataset " + modelname)

        # save outliers name
        outliers_names = y_test[y_pred == 5]['official_name']
        outliers_names.to_csv("../Data/outputs/outliers-testset-" + modelname + "-onlyonedataset.csv", index=False)

        # print("final score : %f" % totalscore)
        print("plot")
        cnf_matrix = confusion_matrix(y_test['label'].astype('category').cat.codes, y_pred)
        # plt.figure(figsize=(10, 10))
        # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
        print()
        np.set_printoptions(precision=2)
        # PlotDir non-normalized confusion matrix
        plt.figure.Figure(figsize=(10, 10))

        plot_confusion_matrix(cnf_matrix,
                              title="with-unknown-testset-" + modelname + "--onlyonedataset",
                              classes=names.append(pd.Index(["Unknown"])))
        with open("../Data/outputs/with-unknown-testset-" + modelname + "-" + "-onlyonedataset.txt", 'w') as f:
            print(classification_report(y_test['label'].astype('category').cat.codes, y_pred, ), file=f)
        models.append(clf)
    return models, modelnames


def test(x_transformed, y2, models, modelnames, names):
    for clf, modelname in zip(models, modelnames):
        if modelname == "bnn":
            # test bnn on stomach
            tot, correct_predictions, predicted_for_images, new_prediction, probabilities = clf.test_batch(
                torch.from_numpy(x_transformed).float(), y2, names, plot=False)
            y_pred = new_prediction
            maxprob = np.max(probabilities, axis=1)

        elif modelname == "mlp":
            # test bnn on stomach
            probabilities, true_labels = clf.test_forced(x_transformed, y2)
            y_pred = np.argmax(probabilities, axis=1)
            maxprob = np.max(probabilities, axis=1)
            y_pred[maxprob < 0.9] = 5

        elif modelname == "mlptree":
            # test bnn on stomach
            maxprob, y_pred, true_labels, probabilities = clf.test_forced(x_transformed, y2)
            y_pred[maxprob < 0.9] = 5
        else:
            return
        # save results

        df = pd.DataFrame({
            'official_name': y2['official_name'].tolist(),
            'max_probability': maxprob.tolist(),
            'probabilities': np.array(probabilities).tolist(),
            'y_pred': y_pred.tolist(),
            'y_true': y2["label"].astype('category').cat.codes.tolist()
        })
        PlotInstograms(df, "istogramma stomaco unico dataset " + modelname)
        df.to_csv("../Data/outputs/pred-stomaco-" + modelname + "-" + "-onlyonedataset.csv")
        outliers_names = y2[y_pred == 5]['official_name']
        print(outliers_names)
        outliers_names.to_csv("../Data/outputs/outliers-stomaco-" + modelname + "-" + "-onlyonedataset.csv",
                              index=False)
        y_pred = y_pred.astype(np.float)
        y_true = y2['label'].astype('category').cat.codes
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
                              title="with-unknown-stomaco-" + modelname + "--onlyonedataset",
                              classes=["predicted", "unknown"])

        with open("../Data/outputs/with-unknown-stomaco-" + modelname + "-" + "-onlyonedataset.txt", 'w') as f:
            print(classification_report(y2['label'].astype('category').cat.codes, y_pred, ), file=f)
