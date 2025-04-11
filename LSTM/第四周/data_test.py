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
device = torch.device("cpu")

from plot_results import plot_results

def address_data(data_train):
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

    data_frames = {sheet: df for sheet, df in data_frames.items() if df is not None}

    df_train = []
    for sheet, df in data_frames.items():
        if len(df.columns[df.isnull().any()].tolist()) == 0:
            df_train.append(df)

    X_df, Y_df = address_data(df_train)

    return X_df, Y_df


# 准备数据
X_list, Y_list = load_data('/Users/poria/git/Graduation-project/LSTM/第三周/bio_train_new.xlsx')
X_test_list, Y_test_list = load_data('/Users/poria/git/Graduation-project/LSTM/第三周/bio_test.xlsx')

# 归一化数据
scaler_X = MinMaxScaler()
scaler_Y1 = MinMaxScaler()
scaler_Y2 = MinMaxScaler()

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
    other_features = np.array(scaler_X.fit_transform(X.drop(columns=['发酵周期/h','转速r/min','风量L/h'])), dtype=np.float32)
    # 合并特征
    X_values = np.concatenate([time_feature,air_flow,rotation_speed,other_features], axis=1)
    # 转换为PyTorch张量
    X_ls.append(torch.tensor(X_values, dtype=torch.float32))

# 填充序列，使它们具有相同的长度
X_padded_sequences = pad_sequence(X_ls, batch_first=True, padding_value=-1)
lengths_train = torch.tensor([len(x) for x in X_ls])

# 对训练集Y进行归一化
Y_values1 = np.concatenate([Y.iloc[:, 0].values.flatten().reshape(-1, 1) for Y in Y_list], axis=0)
Y_values2 = np.concatenate([Y.iloc[:, 1].values.flatten().reshape(-1, 1) for Y in Y_list], axis=0)
scaler_Y1.fit(Y_values1)
scaler_Y2.fit(Y_values2)

y_ls = [torch.tensor(np.hstack([scaler_Y1.transform(Y.iloc[:, 0].values.flatten().reshape(-1, 1)),
                                scaler_Y2.transform(Y.iloc[:, 1].values.flatten().reshape(-1, 1))]), dtype=torch.float32) for Y in Y_list]
y_padded_sequences = pad_sequence(y_ls, batch_first=True, padding_value=-1)
y_train = y_padded_sequences

# # 对测试集X进行归一化

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
    other_features = np.array(scaler_X.transform(X.drop(columns=['发酵周期/h','转速r/min','风量L/h'])), dtype=np.float32)
    # 合并特征
    X_values = np.concatenate([time_feature,air_flow,rotation_speed,other_features], axis=1)
    # 转换为PyTorch张量
    X_test_ls.append(torch.tensor(X_values, dtype=torch.float32))

# 填充序列，使它们具有相同的长度
X_test_padded_sequences = pad_sequence(X_test_ls, batch_first=True, padding_value=-1)
lengths_test = torch.tensor([len(x) for x in X_test_ls])




# 对测试集Y进行归一化
y_test_ls = [torch.tensor(np.hstack([scaler_Y1.transform(Y.iloc[:, 0].values.flatten().reshape(-1, 1)),
                                     scaler_Y2.transform(Y.iloc[:, 1].values.flatten().reshape(-1, 1))]), dtype=torch.float32) for Y in Y_test_list]
y_test_padded_sequences = pad_sequence(y_test_ls, batch_first=True, padding_value=-1)
y_test = y_test_padded_sequences

# 创建数据加载器
train_dataset = TensorDataset(X_padded_sequences, lengths_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = TensorDataset(X_test_padded_sequences, lengths_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 假设我们有两个输出值

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))  # 使用total_length参数
        # 对每个时间步进行预测
        out = self.fc(output)
        return out

input_size = X_padded_sequences.shape[2]
hidden_size = 128
num_layers = 3
output_size = 2  # 设置输出大小为2
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)  # 每50个epoch学习率衰减为原来的0.9倍

# 训练模型
num_epochs = 300
best_loss = 99999
best_eval_loss = 99999

for epoch in range(num_epochs):
    model.train()
    loss_epoch = 0
    for inputs, lengths, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        lengths = lengths.cpu().to(torch.int64)  # 确保 lengths 在 CPU 上并且是 int64 类型
        outputs = model(inputs, lengths)

        # 计算所有时间步的损失，忽略填充部分的损失
        mask = (inputs[:, :, 0] != -1).float().unsqueeze(-1).to(device)
        loss = (criterion(outputs, targets) * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_epoch += loss.item()
    if optimizer.state_dict()['param_groups'][0]['lr'] > 2e-5:
        scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_epoch/len(train_loader):.4f}')
    print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])
    if loss_epoch < best_loss:
        best_loss = loss_epoch/len(train_loader)

    # 保存最佳模型
    if epoch == 0 or loss_epoch < best_loss:
        torch.save(model.state_dict(), 'model.pt')

    # 加载并且评估
    model_state_dict = torch.load('model.pt', map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()

    with torch.no_grad():
        loss_eval = 0
        all_preds = []
        all_targets = []

        for inputs, lengths, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            lengths = lengths.cpu().to(torch.int64)  # 确保 lengths 在 CPU 上并且是 int64 类型
            outputs = model(inputs, lengths)

            # 计算所有时间步的损失，忽略填充部分的损失
            mask = (inputs[:, :, 0] != -1).float().unsqueeze(-1).to(device)
            loss = (criterion(outputs, targets) * mask).sum() / mask.sum()
            loss_eval += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

        avg_eval_loss = loss_eval / len(test_loader)
        print(f'Eval Loss: {avg_eval_loss:.4f}')

        # 将预测值和真实值从填充值中提取出来
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        preds_inverse = np.zeros_like(all_preds)
        targets_inverse = np.zeros_like(all_targets)
        for i in range(all_preds.shape[0]):
            for j in range(all_preds.shape[1]):
                preds_inverse[i, j, 0] = scaler_Y1.inverse_transform(all_preds[i, j, 0].reshape(-1, 1)).squeeze()
                preds_inverse[i, j, 1] = scaler_Y2.inverse_transform(all_preds[i, j, 1].reshape(-1, 1)).squeeze()
                targets_inverse[i, j, 0] = scaler_Y1.inverse_transform(all_targets[i, j, 0].reshape(-1, 1)).squeeze()
                targets_inverse[i, j, 1] = scaler_Y2.inverse_transform(all_targets[i, j, 1].reshape(-1, 1)).squeeze()

        # 去除填充部分
        valid_mask = all_targets[:, :, 0] > 0

for i in range(2):  # 对于两个输出值
    true_values = targets_inverse[:, :, i][valid_mask]
    predicted_values = preds_inverse[:, :, i][valid_mask]
    attribute_name = Y_list[0].columns[i]
    plot_results(true_values, predicted_values, attribute_name[:2])