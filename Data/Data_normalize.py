from os import listdir

import pandas as pd
from scipy import stats


def union(row):
    return str(row["is_tumor"]) + "-" + row["project_id"].split("-")[1]


def map(row):
    return str(row["case_id"]) + "_" + str(row["is_tumor"])


max_features = 5000
fromroot = "./original"
toroot = "./data"

for directory in listdir(fromroot):
    for filename in listdir(fromroot + "/" + directory):
        frompath = fromroot + "/" + directory + "/" + filename
        topath = fromroot + "/" + directory + "/" + filename
        preprocesspath = fromroot + "/" + directory + "/preprocessed" + filename

        if filename == "annotation_global.csv":
            data = pd.read_csv(filename, sep='\t')
            data["case_id"] = data.apply(lambda row: map(row), axis=1)
            cases_removed = data[data["project_id"] == "TCGA-SARC"]
            data = data[data["project_id"] != "TCGA-SARC"]
            data["is_tumor"] = data["is_tumor"].map({0: 'healthy', 1: 'tumor'})
            data["label"] = data.apply(lambda row: union(row), axis=1)
            data = data.sort_values(by="case_id")
            data.to_csv(topath, index=False)
        else:
            data = pd.read_csv(frompath)
            data = data.reset_index()
            headers = data.iloc[0]
            data = pd.DataFrame(data.values[1:], columns=headers)
            index = data["Composite Element REF"]
            data_onlyvalues = data.drop(columns=["Composite Element REF"])
            data2 = pd.DataFrame(stats.zscore(data_onlyvalues.values.tolist()))
            data2["Composite Element REF"] = index
            data2 = data2[~data2["Composite Element REF"].isin(cases_removed["case_id"])]
            data2 = data2.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
            data2.to_csv(topath, index=False, header=False)
            # data=data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index]
            data = pd.DataFrame(
                stats.zscore(
                    data[data_onlyvalues.std().sort_values(ascending=False).head(max_features).index].values.tolist()))
            data["Composite Element REF"] = index
            data = data[~data["Composite Element REF"].isin(cases_removed["case_id"])]
            data = data.sort_values(by="Composite Element REF").drop(columns="Composite Element REF")
            data.to_csv(preprocesspath, index=False, header=False)
