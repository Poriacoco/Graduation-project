import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from data_address_wholeTable import load_data
import os
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
device = torch.device("cuda:6")

# 准备数据
X_list, Y_list = load_data('data/bio_train_new.xlsx')
X_test_list, Y_test_list = load_data('data/bio_test_new.xlsx')

# 归一化数据
scaler_X = MinMaxScaler()
scaler_Y1 = MinMaxScaler()
scaler_Y2 = MinMaxScaler()

# 对训练集X进行归一化
X_ls = []
for X in X_list:
    X_values = np.append(np.array(X['发酵周期/h'], dtype=np.float32).reshape(-1, 1)/100,
                         np.array(scaler_X.fit_transform(X.drop(columns=['发酵周期/h'])), dtype=np.float32), axis=1)
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

# 对测试集X进行归一化
X_test_ls = []
for X in X_test_list:
    X_values = np.append(np.array(X['发酵周期/h'], dtype=np.float32).reshape(-1, 1)/100,
                         np.array(scaler_X.transform(X.drop(columns=['发酵周期/h'])), dtype=np.float32), axis=1)
    X_test_ls.append(torch.tensor(X_values, dtype=torch.float32))

X_test_padded_sequences = pad_sequence(X_test_ls, batch_first=True, padding_value=-1)
lengths_test = torch.tensor([len(x) for x in X_test_ls])

# 对测试集Y进行归一化
y_test_ls = [torch.tensor(np.hstack([scaler_Y1.transform(Y.iloc[:, 0].values.flatten().reshape(-1, 1)),
                                     scaler_Y2.transform(Y.iloc[:, 1].values.flatten().reshape(-1, 1))]), dtype=torch.float32) for Y in Y_test_list]
y_test_padded_sequences = pad_sequence(y_test_ls, batch_first=True, padding_value=-1)
y_test = y_test_padded_sequences

# 创建数据加载器
train_dataset = TensorDataset(X_padded_sequences, lengths_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

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
hidden_size = 64
num_layers = 3
output_size = 2  # 设置输出大小为2
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)  # 每100个epoch学习率衰减为原来的0.9倍

# # 训练模型
# num_epochs = 800
# best_loss = 99999
# best_eval_loss = 99999

# for epoch in range(num_epochs):
#     model.train()
#     loss_epoch = 0
#     for inputs, lengths, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         lengths = lengths.cpu().to(torch.int64)  # 确保 lengths 在 CPU 上并且是 int64 类型
#         outputs = model(inputs, lengths)

#         # 计算所有时间步的损失，忽略填充部分的损失
#         mask = (inputs[:, :, 0] != -1).float().unsqueeze(-1).to(device)

#         loss = (criterion(outputs, targets) * mask).sum() / mask.sum()

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#         optimizer.step()
#         loss_epoch += loss.item()
#     if optimizer.state_dict()['param_groups'][0]['lr'] > 2e-5:
#         scheduler.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_epoch/len(train_loader):.4f}')
#     print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])
#     if loss_epoch < best_loss:
#         best_loss = loss_epoch/len(train_loader)

#     # 保存最佳模型
#     if epoch == 0 or loss_epoch < best_loss:
#         torch.save(model.state_dict(), 'model.pt')

# 评估模型并绘制结果
def plot_results(true_values, predicted_values, attribute_name):
    if attribute_name == '残糖':
        attribute_name = 'Residual Sugar'
    else:
        attribute_name = 'Sodium Acid'
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, 'o', label=f'Target values {attribute_name}', linestyle='-', markersize=5)
    plt.plot(predicted_values, 'x', label=f'Predicted values {attribute_name}', linestyle='--', markersize=5)
    plt.xlabel('Sample Index')
    plt.ylabel(attribute_name)
    plt.legend()
    plt.title(f'Contrast between Target and Prediction: {attribute_name}')
    plt.savefig(f'pic_{attribute_name}.png')


# 加载并评估模型
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