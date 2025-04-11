import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import KFold


def address_data(data_all):
    """
    生成target滞后特征，并划分自变量，因变量
    """
    X1, Y1 = [], []
    for data in data_all:
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


    df_all = []
    for sheet, df in data_frames.items():
        if len(df.columns[df.isnull().any()].tolist()) == 0:
            df_all.append(df)


    # 初始化KFold（按sheet索引划分）
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    sheet_indices = np.arange(len(df_all))  # [0, 1, 2]

    for fold_idx, (train_sheet_idx, val_sheet_idx) in enumerate(kfold.split(sheet_indices)):
        # 获取当前fold的训练集和验证集的sheet数据
        df_train = [df_all[i] for i in train_sheet_idx]
        df_test = [df_all[i] for i in val_sheet_idx]


    X_df, Y_df = address_data(df_train)
    X_test_df, Y_test_df=address_data(df_test)
    return X_df, Y_df,X_test_df, Y_test_df




























scaler_X = MinMaxScaler()
scaler_Y1 = MinMaxScaler()
scaler_Y2 = MinMaxScaler()

def scaler_train_data(X_list):
    max_period_train = []
    for X in X_list:
        max_period_train.append(X['发酵周期/h'].max())

    X_ls = []
    for i, X in enumerate(X_list):
        # 处理时间特征
        time_feature = np.array(X['发酵周期/h'], dtype=np.float32).reshape(-1, 1) / max_period_train[i]
        # 处理风量列
        air_flow = np.array(X['风量L/h'], dtype=np.float32).reshape(-1, 1) / 10000
        # 处理转速列
        rotation_speed = np.array(X['转速r/min'], dtype=np.float32).reshape(-1, 1) / 1000
        # 标准化其他特征
        other_features = np.array(scaler_X.fit_transform(X.drop(columns=['发酵周期/h', '转速r/min', '风量L/h'])),
                                  dtype=np.float32)
        # 合并特征
        X_values = np.concatenate([time_feature, air_flow, rotation_speed, other_features], axis=1)
        # 转换为PyTorch张量
        X_ls.append(torch.tensor(X_values, dtype=torch.float32))

    return X_ls


def scaler_test_data(X_test_list):
    max_period_test = []
    for X in X_test_list:
        max_period_test.append(X['发酵周期/h'].max())

    X_test_ls = []
    for i, X in enumerate(X_test_list):
        # 处理时间特征
        time_feature = np.array(X['发酵周期/h'], dtype=np.float32).reshape(-1, 1) / max_period_test[i]
        # 处理风量列
        air_flow = np.array(X['风量L/h'], dtype=np.float32).reshape(-1, 1) / 10000
        # 处理转速列
        rotation_speed = np.array(X['转速r/min'], dtype=np.float32).reshape(-1, 1) / 1000
        # 标准化其他特征
        other_features = np.array(scaler_X.transform(X.drop(columns=['发酵周期/h', '转速r/min', '风量L/h'])),
                                  dtype=np.float32)
        # 合并特征
        X_values = np.concatenate([time_feature, air_flow, rotation_speed, other_features], axis=1)
        # 转换为PyTorch张量
        X_test_ls.append(torch.tensor(X_values, dtype=torch.float32))
    return X_test_ls
