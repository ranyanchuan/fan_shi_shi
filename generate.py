from Dataloader import Hour
import pandas as pd
from utils import chatGLM, llm_res_json
import numpy as np
from CreateCFsWithGLT import Creator
from Encoders.TabularEncoder import TabEncoder

from category_encoders import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from vali_model import Net


import argparse

## 设定参数，包含原型样本选取规则、局部特征筛选规则、数据集、局部贪心树深度、gpu设备、验证模型、目标类别及样本类别
parse = argparse.ArgumentParser()
# 原型样本选取规则
parse.add_argument("-p", "--proto", choices=["near","rep","cen", "cos", "good"], default="rep",
                   help="Choosing a selection methods of prototype sample(s) which are used to lead the counterfactual generation.\
                    You can separately 'near', 'rep'& 'cent'. 'near' means that selecting the nearest sample as the prototype sample,\
                    'rep' means that selecting the sample which have the highest possibility as the most representative sample,\
                    'cen' means that selecting the sample located in the center,\
                    'ncos' means that selecting the sample which have the highest cosine similarity.")
# 局部特征筛选规则
parse.add_argument("-f", "--func", choices=["fcs","ncs","rss"], default="rss",
                   help="Choosing a criteria to evaluate local feature combination.\
                   You can select 'lrs','pxm', 'lrs' indicates local relative similarity, 'pxm' means proximity.")
# 数据集
parse.add_argument("-d", "--data", choices=["hour"], default="hour", help="Choosing a dataset.")
# 树深度
parse.add_argument("-dp", "--depth", required=True, type=int, help="Setting the depth of local greedy tree.")
# gpu 个数
parse.add_argument("-g", "--gpu", type=int,default=-1, help="Selecting gpu number.")
# 检验模型
parse.add_argument("-vm", "--vali_model", choices=["RF", "NB", "DT", "SVM", "MLP", "KNN"], default="RF")
# 目标类别
parse.add_argument("-ri", "--row_index", required=True, type=int, help="样本行索引")

# 生成反事实个数
parse.add_argument("-n", "--n_ces", type=int, default=3, help="Numbers of CEs.")
args = parse.parse_args()


## 定义模型，选取验证模型
RF = RandomForestClassifier()
NB = GaussianNB()
DT = DecisionTreeClassifier()
SVM = SVC()
MLP = MLPClassifier(max_iter=1000)
KNN = KNeighborsClassifier()

# Model initialization.
print("========== validation model initialization... ==========")
v_model = RF
others = []
if args.vali_model == "RF":
    v_model = RF
    others = [NB, DT, SVM, MLP, KNN]
    pass
elif args.vali_model == "NB":
    v_model = NB
    others = [RF, DT, SVM, MLP, KNN]
    pass
elif args.vali_model == "DT":
    v_model = DT
    others = [RF, NB, SVM, MLP, KNN]
    pass
elif args.vali_model == "SVM":
    v_model = SVM
    others = [RF, NB, DT, MLP, KNN]
    pass
elif args.vali_model == "MLP":
    v_model = MLP
    others = [RF, NB, DT, SVM, KNN]
    pass
elif args.vali_model == "KNN":
    v_model = KNN
    others = [RF, NB, DT, SVM, MLP]
    pass
print("========== model created. ==========")

## 确定类别属性
consum = "" # 反事实目标列
root_path = ""

## 导入数据
print("========== data loading... ==========")
global _data,data
if args.data == "hour":
    _data = Hour()
    data = _data.load_data()
    consum = "high_consum" # 反事实目标列
    root_path = "Hour/"

root_path += "effi/"
root_path="./"+root_path


X = _data.data
y = _data.target
encoder = TabEncoder(data,_data.categoric)
X = encoder.encode(X)
print("========== data loaded. ==========")

## 训练模型，设置验证模型
v_model.fit(X, y)
for each in others:
    each.fit(X ,y)
    pass

print("========== verification model initialization... ==========")

dump(v_model,root_path + "v_model.joblib")
model = Net(root_path + "v_model.joblib")

print("========== verification model created. ==========")

## 划分样本
row_index=args.row_index  # 样本下标

sample_row_data = data.iloc[row_index-1:row_index]

# 判断样本的目标值是 1 还是 0
s_label=int(sample_row_data[consum])
d_label= 0 if s_label==1 else 1

# 划分样本
samples = data[data[consum] == s_label]
protos = data[data[consum] == d_label]

# 生成反事实个数
n_ces = args.n_ces

# 表头
header_row = sample_row_data.columns.tolist() # 样本表头
input_data = sample_row_data.iloc[0].values # 样本输入数据

# 保存测试样本
sample_row_data.to_csv(root_path + "samples.csv",index=False)
creator = Creator(model, data, sample_row_data, _data.categorical_features, args.depth, d_label, n_ces)
fixed_col_arr=["gen","Dishwasher"] # 固定列


# a
args.func = "fcs"
args.proto = "good"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func,fixed_col_arr=fixed_col_arr), columns=data.columns.values)            # 默认german,cos,fcs a
CEs.to_csv(root_path + "hour_good_fcs.csv",index=False)
llm_res_json(CEs,header_row,consum,input_data,"hour_good_fcs") # 大模型总结


# b
args.func = "ncs"
args.proto = "near"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func,fixed_col_arr=fixed_col_arr), columns=data.columns.values)
CEs.to_csv(root_path + "hour_near_ncs.csv",index=False)
llm_res_json(CEs,header_row,consum,input_data,"hour_near_ncs") # 大模型总结


#  c
args.func = "rss"
args.proto = "rep"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func,fixed_col_arr=fixed_col_arr), columns=data.columns.values)
CEs.to_csv(root_path + "hour_rep_rss.csv",index=False)
llm_res_json(CEs,header_row,consum,input_data,"hour_rep_rss") # 大模型总结

# # d
args.func = "rss"
args.proto = "cos"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func,fixed_col_arr=fixed_col_arr), columns=data.columns.values)
CEs.to_csv(root_path + "hour_cos_rss.csv",index=False)
llm_res_json(CEs,header_row,consum,input_data,"hour_cos_rss") # 大模型总结

# # e
args.func = "rss"
args.proto = "cen"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func,fixed_col_arr=fixed_col_arr), columns=data.columns.values)
CEs.to_csv(root_path + "hour_cen_rss.csv",index=False)
llm_res_json(CEs,header_row,consum,input_data,"hour_cen_rss") # 大模型总结
