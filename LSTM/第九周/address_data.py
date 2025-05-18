import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import KFold

def load_data(file_path):
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

    # 判断空值
    df_all = []
    for sheet, df in data_frames.items():
        if len(df.columns[df.isnull().any()].tolist()) == 0:
            df_all.append(df)

    full_data = []
    for data in df_all:
        # 为指定列生成滞后特征
        for column in ['酸钠', '残糖g/dl']:
           data[f'{column}_next'] = data[column].shift(-1)
        data=data.drop(data.index[-1])
        full_data.append(data)

    return full_data


#

# 读取Excel文件
def add_features(file_path = "bio_all_34.xlsx"):
    remove_features = [
        "碱重kg",  # 原始特征
        "重量KG",  # 原始特征
        "PH值_差值",  # 差值特征
        "罐压_差值",  # 差值特征
        "风量L/h_差值",  # 差值特征
        "转速r/min_差值",  # 差值特征
        "温度_差值"  # 差值特征
    ]

    sheets = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in sheets.items():
        # 设置第一列为索引（特征名）
        df = df.set_index(df.columns[0])
        # 计算差值（沿列方向，即行内相邻单元格）
        diff_df = df.diff(axis=1).iloc[:, 1:]

        # 生成新行名并合并到原DataFrame
        new_rows = []
        for feature in df.index:
            diff_row = pd.Series(
                diff_df.loc[feature].values,
                index=df.columns[1:],  # 差值从第二列开始
                name=f"{feature}_差值"
            )
            new_rows.append(diff_row)

        # 合并原数据和新行
        updated_df = pd.concat([df, pd.DataFrame(new_rows)])

        # 删除指定行（包括原始特征和差值特征）
        updated_df = updated_df.drop(remove_features, errors="ignore")
        updated_df = updated_df.fillna(0)

        # 重置索引并将特征名恢复为列
        updated_df = updated_df.reset_index().rename(columns={"index": "特征"})

        # 更新当前工作表数据
        sheets[sheet_name] = updated_df
    # 保存到新Excel文件
    with pd.ExcelWriter("bio_all_34_new2.xlsx") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)







