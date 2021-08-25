from os import listdir
import pandas as pd
from scipy import stats
import numpy as np

def union(row):
    return str(row["is_tumor"]) + "-" + row["project_id"].split("-")[1]


def map(row):
    return str(row["case_id"]) + "_" + str(row["is_tumor"])


max_features = 10000
fromroot = "./original"
toroot = "./data"

for directory in listdir(fromroot):
    if directory== "kidney":
        ignore=False
        for filename in listdir(fromroot + "/" + directory):
            frompath = fromroot + "/" + directory + "/" + filename
            topath = toroot + "/" + directory + "/" + filename
            preprocesspath = toroot + "/" + directory + "/preprocessed" + filename
            if filename == "annotation_global.csv":
                data = pd.read_csv(frompath, sep='\t')
                data["case_id"] = data.apply(lambda row: map(row), axis=1)
                cases_removed = data[data["project_id"] == "TCGA-SARC"]
                data = data[data["project_id"] != "TCGA-SARC"]
                data["is_tumor"] = data["is_tumor"].map({0: 'healthy', 1: 'tumor'})
                data["label"] = data.apply(lambda row: union(row), axis=1)
                data = data.sort_values(by="case_id")
                data.to_csv(topath, index=False)
                continue
            elif filename == "Matrix_miRNA_deseq_correct.csv":
                file = pd.read_csv(frompath, sep=',').transpose()
            elif filename == "Matrix_mRNA_deseq_normalized_prot_coding_correct.csv":
                file = pd.read_csv(frompath, sep=',').transpose()
            elif filename == "Matrix_meth.csv":
                file = pd.read_csv(frompath)
                features= file.columns
                data2 = file.drop(columns="Composite Element REF")
                data2.to_csv(topath, index=False)
                features_preprocessed = data2.std().sort_values(ascending=False).head(max_features).index
                data3= data2[features_preprocessed]
                data3.to_csv(preprocesspath, index=False)
                pd.DataFrame(features.to_numpy()).to_csv(toroot + "/" + directory + "/features" + filename)
                pd.DataFrame(features_preprocessed.to_numpy()).to_csv(toroot + "/" + directory + "/prepfeatures" + filename)
            else:
                continue
            if filename != "Matrix_meth.csv":
                data = file.reset_index()
                headers = data.iloc[0]
                headers["index"] = "Composite Element REF"
                data = pd.DataFrame(data.values[1:], columns=headers)
                features=data.columns
                index = data["Composite Element REF"]
                data_onlyvalues = data.drop(columns=["Composite Element REF"])
                data2 = pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
                data2["Composite Element REF"] = index
                data2 = data2[~data2["Composite Element REF"].isin(cases_removed["case_id"])]
                data2 = data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
                data2.to_csv(topath, index=False)
                # data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
                features_preprocessed=data_onlyvalues.std().sort_values(ascending=False).head(max_features).index

                data = pd.DataFrame(
                    stats.zscore(
                        data[features_preprocessed].values.tolist()))
                data["Composite Element REF"] = index
                data = data[~data["Composite Element REF"].isin(cases_removed["case_id"])]
                data = data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
                data.to_csv(preprocesspath, index=False)
                pd.DataFrame(features.to_numpy()).to_csv(toroot + "/" + directory + "/features" + filename)
                pd.DataFrame(features_preprocessed.to_numpy()).to_csv(toroot + "/" + directory + "/prepfeatures" + filename)