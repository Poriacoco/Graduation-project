import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.ops.numpy_ops import array
from torch.utils.data import DataLoader, TensorDataset
import os
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from sklearn.model_selection import KFold


device = torch.device("cpu")

from plot_results import plot_results
from plot_results import loss_plot
from address_data import load_data
from address_data import scaler_train_data
from address_data import scaler_test_data




def address_data(data_train):
    """
    生成target滞后特征，并划分自变量，因变量
    """
    X1, Y1 = [], []
    for data in data_train:
        # 为指定列生成滞后特征
        for column in ['酸钠', '残糖g/dl']:
           data[f'{column}_next'] = data[column].shift(-1)

        data=data.drop(data.index[-1])
        # 划分自变量（排除_next列）和因变量
        exog_columns = [col for col in data.columns if not col.endswith('_next')]
        exog_data = data[exog_columns].dropna()
        endog_data = data.loc[exog_data.index, ['酸钠_next', '残糖g/dl_next']]

        X1.append(exog_data)
        Y1.append(endog_data)

    return X1, Y1


def load_data(file_path,obj='酸钠'):
    # 获取所有sheet的名称
    sheet_names = pd.ExcelFile(file_path).sheet_names
    # 读取所有sheet并存储在一个字典中
    data_frames = {}
    for sheet in sheet_names:
        data_frames[sheet] = pd.read_excel(file_path, sheet_name=sheet)

    for sheet, df in data_frames.items():
        # 行列互换（转置）
        df = df.transpose()

        # 将转置后的DataFrame的第一行设为列名
        df.columns = df.iloc[0]
        df = df[1:]

        df.reset_index(inplace=True, drop=False)
        df.rename(columns={'index': '发酵周期/h'}, inplace=True)
        df.columns.name = sheet

        # 转换为数值类型，如果不能转换则设置为NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # 更新字典中的 DataFrame
        if len(df.index) != 0:
            data_frames[sheet] = df
        else:
            data_frames[sheet] = None

    # 处理缺失值
    # data_frames = {sheet: df for sheet, df in data_frames.items() if df is not None}
    # df_train = []
    # for sheet, df in data_frames.items():
    #     if len(df.columns[df.isnull().any()].tolist()) == 0:
    #         df_train.append(df)










    all_sheets =data_frames

    # 初始化KFold（按sheet索引划分）
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    sheet_indices = np.arange(len(all_sheets))  # [0, 1, 2]

    for fold_idx, (train_sheet_idx, val_sheet_idx) in enumerate(kfold.split(sheet_indices)):
        # 获取当前fold的训练集和验证集的sheet数据
        train_sheets = [all_sheets[i] for i in train_sheet_idx]
        val_sheets = [all_sheets[i] for i in val_sheet_idx]

        # 合并数据（假设模型需要整体训练）
        X_train = pd.concat([df[['feature1', 'feature2']] for df in train_sheets])
        y_train = pd.concat([df['target'] for df in train_sheets])
        X_val = pd.concat([df[['feature1', 'feature2']] for df in val_sheets])
        y_val = pd.concat([df['target'] for df in val_sheets])

        print(f"Fold {fold_idx + 1}")
        print(f"  训练集: {len(train_sheets)}个sheet, 样本数={X_train.shape[0]}")
        print(f"  验证集: {len(val_sheets)}个sheet, 样本数={X_val.shape[0]}\n")

    X_df, Y_df = address_data(df_train)

    return X_df, Y_df
