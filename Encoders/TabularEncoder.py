from sklearn.preprocessing import StandardScaler,normalize
import pandas as pd
from category_encoders import *


class TabEncoder():
    def __init__(self,dataset:pd.DataFrame,categorical_features:list):
        self.X = dataset.iloc[:,:-1]
        self.y = dataset.iloc[:,-1]
        self.scaler = StandardScaler()
        self.categorical_features = categorical_features
        self.encoder = TargetEncoder(cols=self.categorical_features).fit(self.X,self.y)
        X = self.encoder.transform(self.X)
        self.scaler.fit(X)
        pass
    def encode(self,en):
        numeric_data = self.encoder.transform(en)
        numeric_data = self.scaler.transform(numeric_data)
        numeric_data = normalize(numeric_data)
        return numeric_data
    pass


