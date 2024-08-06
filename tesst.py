import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import torch.nn.functional as F
import torch.nn as nn
import tqdm

train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
label = train.iloc[:,-1]
train = train.iloc[:,:-1]

#数据预处理
all_data = pd.concat([train,test])
#标准化
numerical_features=all_data.select_dtypes(exclude=['object']) #得到不含object类型数据的DataFrame
mean = numerical_features.mean()
std = numerical_features.std()
numerical_features=(numerical_features - mean) / std
numerical_features.fillna(0,inplace=True)
all_data.loc[:,numerical_features.columns] = numerical_features
#one-hot
all_data=pd.get_dummies(all_data,dummy_na=True,dtype=int)

#数据集划分
num_cv = 120
train_features = torch.tensor(all_data.iloc[:train.shape[0]-num_cv,:].values,dtype=torch.float32,device="cuda")
cv_features = torch.tensor(all_data.iloc[train.shape[0]-num_cv:train.shape[0],:].values,dtype=torch.float32,device="cuda")
test_features = torch.tensor(all_data.iloc[train.shape[0]:,:].values,dtype=torch.float32)
train_labels = torch.tensor(label.iloc[:train.shape[0]-num_cv].values,dtype=torch.float32,device='cuda')
cv_labels = torch.tensor(label.iloc[train.shape[0]-num_cv:train.shape[0]].values,dtype=torch.float32,device='cuda')

#生成迭代器

#构建网络

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
net = Net(train_features.shape[1],2)
net.apply(init_weights)
net.to(device='cuda')

#损失和优化

def log_rmse(net,features,labels):
    clipped_preds=torch.clamp(net(features),1,float('inf'))
    rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()
#训练

#预测
def predict(net,test_features,test_data):
    net.to(device = 'cpu')
    test_data = test_data.reset_index()
    y_pred = net(test_features).detach().cpu()
    predictions = torch.round(y_pred).numpy()

    submission.to_csv('submission.csv')

if __name__ == '__main__':
    n = len(tr_loss)
    plt.plot(range(n), tr_loss,label='train')
    plt.plot(range(n), cv_loss, color='green',label='cv')
    plt.legend()
    plt.show()
    predict(net,test_features,test)
