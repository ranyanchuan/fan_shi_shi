import pandas as pd
import numpy as np
from GLT import LinkTree
from tqdm import tqdm
from ExtractProto import GetPrototypes
import torch
from Encoders.TabularEncoder import TabEncoder

class Creator(object):
    
    def __init__(self,model,datas:pd.DataFrame,samples:pd.DataFrame,categorical_features,deepth,target_label,n_CFs):
        self.model = model
        self.target_label = target_label
        self.goals = datas[datas.iloc[:,-1] == self.target_label]
        self.samples = samples
        self.deepth = deepth
        self.n_CFs = n_CFs
        self.counterfactuals = []
        self.categorical_features = categorical_features
        self.encoder = TabEncoder(datas,self.categorical_features)
        pass



    def createCFs(self, way, dataset, device,func,fixed_col_arr):
     
        # todo 使用GPU 加速   device
        device = torch.cuda.set_device(-1) # 不使用 GPU 加速
        Prototype = GetPrototypes(self.model, self.goals, self.categorical_features, self.n_CFs, self.target_label, self.encoder)  # model用于提取原型；goals为原型类别的样本；n_CFS为生成反事实的数量，也即提取的原型数，对应该类中的n_protos；target_label为目标类别，也就是原型样本的类别
        protos,indices = Prototype.get_protos(way, self.samples, dataset)
        # print("protos===>>>>",protos)
        # 保存原始数据中的原型
        in_list = indices.data.cpu().numpy().tolist()

        # 原型数据   
        original_protos = self.goals.iloc[in_list].copy()
        protos = pd.DataFrame(protos,columns=self.goals.columns.values[:-1])

        # 样本数据
        temp_samples=self.samples.iloc[0]
        # 原型数据替换成样本数据，固定列
        for col in fixed_col_arr: 
            original_protos[[col]]= temp_samples[col]
            protos[[col]]= temp_samples[col]
   
        protos = self.encoder.encode(protos)
        samples = self.encoder.encode(self.samples.iloc[:,:-1])
      
        protos = np.array(protos, dtype='float')
        samples = np.array(samples)

        # 转化为Tensor
        protos = torch.tensor(protos,dtype=torch.float).to(device) # 原型   46,Self-Employed,Masters,Married,Sales,White,1,50,1
        samples = torch.tensor(samples,dtype=torch.float).to(device) # 样本   Self-Employed,Masters,Married,Sales,White,1,50,1
        count = 0

        # 笛卡尔积
        for sample in tqdm(samples):
            reckoning = 0
            for proto in protos:
                CF = []
                tree = LinkTree(self.deepth,proto,sample)
                _,counterfactual_path = tree.create_CF(func) # 求树的最优解

                for i in range(len(counterfactual_path)):
                    if counterfactual_path[i] == 0:
                        CF.append(self.samples.iloc[count, i])
                        pass
                    else:
                        CF.append(original_protos.iloc[reckoning, i])
                        pass
                    pass

                counterfactual = self.encoder.encode(pd.DataFrame(np.array(CF).reshape(1,-1),columns=self.goals.columns.values[:-1]))    
                counterfactual = torch.tensor(np.array(counterfactual.tolist(),dtype='float')).to(device)
                if self.model.predict(counterfactual) == self.target_label:
                    self.counterfactuals.append(tuple(CF+[self.target_label]))
                    pass
         
                reckoning += 1
                pass
            count += 1
            pass
        return self.counterfactuals
    pass