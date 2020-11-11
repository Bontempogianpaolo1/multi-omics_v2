import pandas as pd
from scipy import stats

path_annotation = "original/reni/annotation_global.csv"
path_meth = "original/reni/Matrix_meth.csv"
max_features = 5000


def union(row):
    return str(row["is_tumor"]) + "-" + row["project_id"].split("-")[1]


def map(row):
    return str(row["case_id"]) + "_" + str(row["is_tumor"])


data = pd.read_csv(path_annotation, sep='\t')
data["case_id"] = data.apply(lambda row: map(row), axis=1)
cases_removed = data[data["project_id"] == "TCGA-SARC"]
data = data[data["project_id"] != "TCGA-SARC"]
data["is_tumor"] = data["is_tumor"].map({0: 'healthy', 1: 'tumor'})
data["label"] = data.apply(lambda row: union(row), axis=1)
data = data.sort_values(by="case_id")
# data=data.drop(columns=["is_tumor","project_id"])
data.to_csv("./data/preprocessed_annotation_global.csv", index=False)

matrixname = "meth"
data = pd.read_csv("./original/Matrix_" + matrixname + ".csv", sep='\t').transpose()
data = data.reset_index()
headers = data.iloc[0]
data = pd.DataFrame(data.values[1:], columns=headers)
index = data["Composite Element REF"]
data_onlyvalues = data.drop(columns=["Composite Element REF"])
data2 = pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
data2["Composite Element REF"] = index
data2 = data2[~data2["Composite Element REF"].isin(cases_removed["case_id"])]
data2 = data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data2.to_csv("./data/Matrix_" + matrixname + ".csv", index=False, header=False)
# data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
data = pd.DataFrame(
    stats.zscore(data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index].values.tolist()))
data["Composite Element REF"] = index
data = data[~data["Composite Element REF"].isin(cases_removed["case_id"])]
data = data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data.to_csv("./data/preprocessed_Matrix_" + matrixname + ".csv", index=False, header=False)

matrixname = "miRNA_deseq_correct"
data = pd.read_csv("./original/Matrix_" + matrixname + ".csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"] = "Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)
index = data["Composite Element REF"]
data_onlyvalues = data.drop(columns=["Composite Element REF"])
data2 = pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
data2["Composite Element REF"] = index
data2 = data2[~data2["Composite Element REF"].isin(cases_removed["case_id"])]
data2 = data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data2.to_csv("./data/Matrix_" + matrixname + ".csv", index=False, header=False)
# data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
data = pd.DataFrame(
    stats.zscore(data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index].values.tolist()))
data["Composite Element REF"] = index
data = data[~data["Composite Element REF"].isin(cases_removed["case_id"])]
data = data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data.to_csv("./data/preprocessed_Matrix_" + matrixname + ".csv", index=False, header=False)

matrixname = "mRNA_deseq_normalized_prot_coding_correct"
data = pd.read_csv("./original/Matrix_" + matrixname + ".csv", sep=',').transpose()
data = data.reset_index()
headers = data.iloc[0]
headers["index"] = "Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)
index = data["Composite Element REF"]
data_onlyvalues = data.drop(columns=["Composite Element REF"])
data2 = pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
data2["Composite Element REF"] = index
data2 = data2[~data2["Composite Element REF"].isin(cases_removed["case_id"])]
data2 = data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data2.to_csv("./data/Matrix_" + matrixname + ".csv", index=False, header=False)
# data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
data = pd.DataFrame(
    stats.zscore(data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index].values.tolist()))
data["Composite Element REF"] = index
data = data[~data["Composite Element REF"].isin(cases_removed["case_id"])]
data = data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
data.to_csv("./data/preprocessed_Matrix_" + matrixname + ".csv", index=False, header=False)

import numpy as np

seed = 1200
# y
annotation_path = "../Data/data/reni/preprocessed_annotation_global.csv"

# filenames
meth_path = "../Data/data/reni/Matrix_meth.csv"
mRNA_path = "../Data/data/reni/Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/reni/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
y_stomaco = "../Data/data/preprocessed_annotation_global.csv"
files = [meth_path, mRNA_normalized_path, mRNA_path]
filenames = ["meth", "mRNA", "miRNA"]

parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
              'max_iter': [200, 400, 600]}

import numpy as np
import pandas as pd

seed = 1200
# y
annotation_path = "../Data/data/reni/preprocessed_annotation_global.csv"




# filenames
meth_path = "../Data/data/reni/Matrix_meth.csv"
mRNA_path = "../Data/data/reni/Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/reni/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path, mRNA_normalized_path, mRNA_path]
filenames = ["meth", "mRNA", "miRNA"]

parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
              'max_iter': [200, 400, 600]}

true_labels = []
data = []
# read labels
y = pd.read_csv(annotation_path)

names = y["label"].astype('category').cat.categories

for file, filename in zip(files, filenames):
    # read matrices
    X = pd.read_csv(file, index_col=False, header=None)

