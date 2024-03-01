import torch
import torch.nn.functional as F
from joblib import load

class Net(object):
    def __init__(self,model_path:str):
        self.model = load(model_path)
        pass
    def predict(self,X:torch.Tensor):
        self.device = X.device
        self.X = X.cpu().numpy()
        self.prediction = self.model.predict(self.X)
        return torch.tensor(self.prediction,dtype=torch.long).to(self.device)
    def predict_proba(self,X):
        self.device = X.device
        self.X = X.cpu().numpy()
        self.prediction_proba = self.model.predict_proba(self.X)
        return torch.tensor(self.prediction_proba,dtype=torch.float).to(self.device)
    pass

