import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_and_fill(df_train, df_missing, columns):
    # 对每个需要填充的列进行处理
    for column in columns:
        if df_missing[column].isnull().any():
            # 使用随机森林模型
            rf = RandomForestRegressor(n_estimators=100, random_state=0)
            
            # 训练集和目标
            X_train = df_train.drop(columns=[column])
            y_train = df_train[column]

            # 需要填充的部分
            X_missing = df_missing[df_missing[column].isnull()].drop(columns=[column])
            
            # 训练模型
            rf.fit(X_train, y_train)
            
            # 预测缺失值
            predicted_values = rf.predict(X_missing)
            
            # 填充缺失值
            df_missing.loc[df_missing[column].isnull(), column] = predicted_values

    return df_missing


# def save_data_frames_to_excel(data_frames, output_file_path):
#     # 使用 ExcelWriter，以便将多个 DataFrame 写入同一个文件
#     with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
#         # 遍历所有的 DataFrame，并将它们写入不同的工作表
#         for sheet_name, df in data_frames.items():
#             df = df.transpose().reset_index()
#             df.columns = df.iloc[0]
#             df = df[1:]
#             # 写入 DataFrame 到特定的工作表
#             df.to_excel(writer, sheet_name=sheet_name, index=False)


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

        # 插值
        # df = df.interpolate(method='quadratic')
        columns_to_interpolate = ['酶活', '菌浓ml/50ml', '菌浓g/50ml', '碱重kg', '重量KG']
        df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method='linear')

        # # 删除缺失值
        # columns =  [col for col in df.columns]
        # df = df[columns].dropna()

        # 更新字典中的 DataFrame
        if len(df.index) != 0:
            data_frames[sheet] = df
        else:
            data_frames[sheet] = None

    data_frames = {sheet: df for sheet, df in data_frames.items() if df is not None}

    df_train = []
    for sheet, df in data_frames.items():
        if len(df.columns[df.isnull().any()].tolist()) == 0:
            df_train.append(df)

    # df_train = pd.concat(df_train)

    # for sheet, df in df_train.items():
    #     columns_to_fill = df.columns[df.isnull().any()].tolist()
    #     df = train_and_fill(df_train, df, columns_to_fill)
    
    # save_data_frames_to_excel(data_frames, file_path[:-5] + '_new.xlsx')

    X_df, Y_df = address_data(df_train)

    return X_df, Y_df


def address_data(data_train):
    X1 = []
    Y1 = []

    for data in data_train:
        # 为每个属性添加滞后特征
        for column in data.columns:
            # if column != '发酵周期/h':
            data[f'{column}_lag_1'] = data[column].shift(1)

        exog_columns = ['发酵周期/h'] + [col for col in data.columns if 'lag' in col]
        exog_data = data[exog_columns].dropna()
        
        # 5. 定义因变量
        endog_data = data.loc[exog_data.index, ['酸钠', '残糖g/dl']]

        X1.append(exog_data)
        Y1.append(endog_data)

    return X1, Y1

    