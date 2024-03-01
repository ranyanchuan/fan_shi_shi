import numpy as np
import pandas as pd
import os

# PATH = os.path.join("/Users/person/T-COL/Datasets/")
PATH = os.path.join("./Datasets/")

# 能耗数据
class Hour():
    """
    能耗数据，小时采集
    """
    hour = pd.read_csv(PATH + "hour_data_label.csv",header=0,index_col=0)
    hour_data = pd.DataFrame(hour)

    def __init__(self):
        self.data = self.hour_data.iloc[:,:-1] # 第0～倒数第2
        self.target = self.hour_data["high_consum"]
        self.categoric = [] # 类型列
        self.continues = self.hour_data.columns.difference(self.categoric).drop("high_consum").values.tolist()
        self.categorical_features = self.hour_data[self.categoric] # 类型特征
        self.continuous_features = self.hour_data[self.continues] # 文本或数字特征
        pass
    def load_data(self):
        return self.hour_data
    pass

