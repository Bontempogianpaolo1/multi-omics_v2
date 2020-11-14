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


annnotationpath = "./annotation_global.csv"
oldpath = "./Matrix_meth(wrong).csv"
toadjustpath = "./meth_batch_corrected.csv"

data = pd.read_csv(annnotationpath, sep='\t')
data["case_id"] = data.apply(lambda row: map(row), axis=1)
cases_removed = data[data["project_id"] == "TCGA-SARC"]
data = data[data["project_id"] != "TCGA-SARC"]
data["is_tumor"] = data["is_tumor"].map({0: 'healthy', 1: 'tumor'})
data["label"] = data.apply(lambda row: union(row), axis=1)
data = data.sort_values(by="case_id")


oldfile = pd.read_csv(oldpath, sep='\t').transpose()
file = pd.read_csv(toadjustpath, sep=',').transpose()




data = oldfile.reset_index()
headers = data.iloc[0]
headers["index"] = "Composite Element REF"
data = pd.DataFrame(data.values[1:], columns=headers)
data = data[~data["Composite Element REF"].isin(cases_removed["case_id"])]


data = data.sort_values(by="Composite Element REF")
file.insert(loc=0,column="Composite Element REF",value=data["Composite Element REF"].values)
#file["Composite Element REF"]= data["Composite Element REF"]
#features=data.columns
final= pd.DataFrame(file.values,columns=data.columns)


final.to_csv("./Matrix_meth_v2.csv", index=False, header=True)
#data= pd.read_csv("./prova.csv",sep=',')
print("ciao")
