from torch.utils.data import Dataset  
import os
import pandas as pd
import numpy as np 
import torch 

class iris_dataload(Dataset):   #锯承dataset必须要重新下面三个函数：init ,len,gettiem
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path 
        self.transform = transform
 
        assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path) 
        
        #AI的数据处理的常用处理方式：！！！
        #这里的核心处理是：把txt中的数据转成pandas的dataframe,再转成numpy，再转成torch。
        df=pd.read_csv(self.data_path , names=[0,1,2,3,4])
        d={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
        df[4]=df[4].map(d)
        data=df.iloc[:,0:4]
        label=df.iloc[:,4:]

        data=np.array(data) 
        #归一化操作：减去均值，再除以方差
        data = (data - np.mean(data) )/ np.std(data) 
        label=np.array(label) 

        self.data=torch.from_numpy(np.array(data,dtype='float32') )   
        self.label= torch.from_numpy(np.array(label,dtype='int64') ) 

        self.data_num = len(label)  # 存储训练集的所有图片路径
        print("{} images were found in the dataset.".format(self.data_num))

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        self.data = list(self.data)   #通过list做样本封装，便于通过index直接获取单个样本
        self.label = list(self.label)
        return self.data[idx], self.label[idx]


 