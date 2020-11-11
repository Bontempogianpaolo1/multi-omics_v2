import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import Classification.models.bnn as bnn
from utils.Plot import plot_confusion_matrix
from sklearn.metrics import classification_report
from utils.Plot import PlotInstograms
import Classification.models as models

seed = 1200
# y
annotation_path = "../Data/data/kidney/preprocessed_annotation_global.csv"

# filenames
meth_path = "../Data/data/kidney/Matrix_meth.csv"
mRNA_path = "../Data/data/kidney/Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/kidney/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
y_stomaco = "../Data/data/stomach/preprocessed_annotation_global.csv"
files = [meth_path, mRNA_normalized_path, mRNA_path]
filenames = ["meth", "mRNA", "miRNA"]

parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
              'max_iter': [200, 400, 600]}

true_labels = []
data = []
data2 = []
# read labels
y = pd.read_csv(annotation_path)
y2 = pd.read_csv(y_stomaco)
names = y["label"].astype('category').cat.categories
names2 = names.append(pd.Index(["Unknown"]))

for file, filename in zip(files, filenames):
    # read matrices
    X = pd.read_csv(file, index_col=False, header=None)
    X2 = pd.read_csv("../Data/data/stomach/Matrix_" + filename + ".csv", index_col=False, header=None)
    # cut useless features
    max_features = np.min([X.shape[1], X2.shape[1]])
    X = X.iloc[:, X.std().sort_values(ascending=False).head(max_features).index]
    X2 = X2.iloc[:, X2.std().sort_values(ascending=False).head(max_features).index]

    data.append(X)
    data2.append(X2)

# concat data
X = pd.concat([data[0], data[1], data[2]], axis=1)
X2 = pd.concat([data2[0], data2[1], data2[2]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y["label"].astype('category').cat.codes,
                                                    shuffle=False, random_state=seed)
names = y["label"].astype('category').cat.categories


# reduce matrix
#final_samples=
n_components = 21
pca = PCA(n_components=n_components)
X_train_transformed = pca.fit_transform(X_train)

sme = SMOTEENN(random_state=42)
X_res, y_res = sme.fit_sample(X_train_transformed, y_train["label"].to_numpy())
X_new=X_res[681:]
y_res=y_res[681:]
for label in names:
    X_cur= X_new[y_res==label]
    tot=(y_res==label).sum()
    if(tot != 0):
        #print(X_cur)
        df_to_add = pd.DataFrame({
            'official_name': np.ndarray((y_res==label).sum()).tolist(),
            'project_id': np.ndarray((y_res == label).sum()).tolist(),
            'case_id': np.ndarray((y_res==label).sum()).tolist(),
            'label': y_res[y_res==label].tolist()
        })
        df_to_add['is_tumor']=df_to_add['label'].str.split('-').str[0]
        y_train=y_train.append(df_to_add)
        X_train_transformed=np.append(X_train_transformed,X_cur,axis=0)

X_test_transformed = pca.transform(X_test)
X_transformed = pca.transform(X2)

models = [models.bnn.BNN, models.Mlp.MLP, models.Mlptree.MlpTree]
modelnames = ["mlptree", "mlp", "bnn"]
for model, modelname in zip(models, modelnames):
    scores = np.empty([])
    # components = np.empty([])
    variances = np.empty([])
    # train and test bnn
    if modelname == "bnn":
        clf = models.bnn.BNN(n_components, 20, 5)
        clf.train_step(X_train_transformed, y_train)
        tot, correct_predictions, predicted_for_images, new_prediction, probabilities = clf.test_batch(
            torch.from_numpy(X_test_transformed).float(), y_test, names, plot=False)
        y_pred = new_prediction
        maxprob = np.max(probabilities, axis=1)

    elif modelname == "mlp":
        # train and test mlp
        clf = models.Mlp.MLP(n_components, 20, 5)
        clf.train_step(X_train_transformed, y_train)
        probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)
        y_pred = np.argmax(probabilities, axis=1)
        maxprob = np.max(probabilities, axis=1)
        y_pred[maxprob < 0.9] = 5

    elif modelname == "mlptree":
        # train and test mlptree
        clf = models.Mlptree.MlpTree(n_components, 20, 5)
        clf.train_step(X_train_transformed, y_train)
        maxprob, y_pred, true_labels, probabilities = clf.test_forced(X_test_transformed, y_test)
        y_pred[maxprob < 0.9] = 5

    # save results
    df = pd.DataFrame({
        'official_name': y_test['official_name'].tolist(),
        'max_probability': maxprob.tolist(),
        'probabilities': np.array(probabilities).tolist(),
        'y_pred': y_pred.tolist(),
        'y_true': y_test["label"].astype('category').cat.codes.tolist()
    })
    df.to_csv("../Data/outputs/pred-testset-" + modelname + "-" + "-onlyonedataset.csv")
    PlotInstograms(df,"istogramma testset unico dataset "+modelname)

    # save outliers name
    outliers_names = y_test[y_pred == 5]['official_name']
    outliers_names.to_csv("../Data/outputs/outliers-testset-" + modelname + "-onlyonedataset.csv", index=False)

    totalscore = accuracy_score(y_test['label'].astype('category').cat.codes, y_pred)
    scores = np.append(scores, totalscore)
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
                          classes=names2)
    with open("../Data/outputs/with-unknown-testset-" + modelname + "-" + "-onlyonedataset.txt", 'w') as f:
        print(classification_report(y_test['label'].astype('category').cat.codes, y_pred, ), file=f)

    if modelname == "bnn":
        # test bnn on stomach
        tot, correct_predictions, predicted_for_images, new_prediction, probabilities = clf.test_batch(
            torch.from_numpy(X_transformed).float(), y2, names, plot=False)
        y_pred = new_prediction
        maxprob = np.max(probabilities, axis=1)

    elif modelname == "mlp":
        # test bnn on stomach
        probabilities, true_labels = clf.test_forced(X_transformed, y2)
        y_pred = np.argmax(probabilities, axis=1)
        maxprob = np.max(probabilities, axis=1)
        y_pred[maxprob < 0.9] = 5

    elif modelname == "mlptree":
        # test bnn on stomach
        maxprob, y_pred, true_labels, probabilities = clf.test_forced(X_transformed, y2)
        y_pred[maxprob < 0.9] = 5

    # save results

    df = pd.DataFrame({
        'official_name': y2['official_name'].tolist(),
        'max_probability': maxprob.tolist(),
        'probabilities': np.array(probabilities).tolist(),
        'y_pred': y_pred.tolist(),
        'y_true': y2["label"].astype('category').cat.codes.tolist()
    })
    PlotInstograms(df,"istogramma stomaco unico dataset "+modelname)
    df.to_csv("../Data/outputs/pred-stomaco-" + modelname + "-" + "-onlyonedataset.csv")
    outliers_names = y2[y_pred == 5]['official_name']
    print(outliers_names)
    outliers_names.to_csv("../Data/outputs/outliers-stomaco-" + modelname + "-" + "-onlyonedataset.csv",
                          index=False)
    print("plot")
    y_pred = y_pred.astype(np.float)
    y_true = y2['label'].astype('category').cat.codes
    y_true[y_true != 5] = 0
    y_true[y_true == 5] = 1
    y_pred[y_pred != 5] = 0
    y_pred[y_pred == 5] = 1

    cnf_matrix = confusion_matrix(y2['label'].astype('category').cat.codes, y_pred)
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
