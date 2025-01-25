import numpy as np
import torch


def create_multivariate_dataset_withfeatures(dataset,window_size):
    """
    将多变量时间序列转化为能够被LSTM训练和预测的数据【特征法】
    参数：
        dataset:DataFrame,包含特征标签,针对此数据,特征滑窗从索引2开始,最后一列是
        window_size:滑窗窗口的大小
    """
    X,y=[],[]
    for i in range(len(dataset)-window_size):
        #.values讲DataFrame转为numpy数组
        #特征滑窗从索引2开始
        feature=dataset.iloc[i:i+window_size,2:-1].values
        target=dataset.iloc[i+window_size-1,-1]
        X.append(feature)
        y.append(target)
    return torch.FloatTensor(np.array(X,dtype=np.float32)),torch.FloatTensor(np.array(y,dtype=np.float32))


def create_multivariate_dataset_withlabels(dataset,window_size):
    """
    多变量时间序列转化为能够被LSTM训练和预测的数据【带标签和特征】
    参数：
        dataset:DataFrame,包含特征标签,针对此数据,特征滑窗从索引2开始,最后一列是
        window_size:滑窗窗口的大小
    """
    X,y=[],[]
    for i in range(len(dataset)-window_size):
        #.values讲DataFrame转为numpy数组
        #特征滑窗从索引2开始
        features_and_labels=dataset.iloc[i:i+window_size,2:].values #与上面函数对比
        target=dataset.iloc[i+window_size,-1] #与上面函数对比
        X.append(features_and_labels)
        y.append(target)
    return torch.FloatTensor(np.array(X,dtype=np.float32)),torch.FloatTensor(np.array(y,dtype=np.float32))

def create_multivariate_dataset_withmask(dataset,window_size):
    """
    多变量时间序列转化为能够被LSTM训练和预测的数据【带标签和特征】
    参数：
        dataset:DataFrame,包含特征标签,针对此数据,特征滑窗从索引2开始,最后一列是
        window_size:滑窗窗口的大小
    """
    X,y=[],[]
    for i in range(len(dataset)-window_size):
        #.values讲DataFrame转为numpy数组
        #特征滑窗从索引2开始
        features_and_labels=dataset.iloc[i:i+window_size,2:].copy().values #与上面函数对比(为什么要copy)
        target=dataset.iloc[i+window_size-1,-1] #与上面函数对比
        #替换最后一个标签
        features_and_labels[-1,-1]=-999
        X.append(features_and_labels)
        y.append(target)
    return torch.FloatTensor(np.array(X,dtype=np.float32)),torch.FloatTensor(np.array(y,dtype=np.float32))
