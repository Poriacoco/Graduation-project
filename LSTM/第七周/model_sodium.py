import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from openpyxl.styles.builtins import total
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
from plot_results import plot_results,loss_plot
from address_data import load_data
from sklearn.model_selection import KFold


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,total_length=x.size(1))
        # 应用Dropout
        output = self.dropout(output)
        # 对每个时间步进行预测
        out = self.fc(output)
        return out



def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, lengths, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # target_sodium, target_sugar = targets.split(1, dim=2)
        target_sodium=targets
        lengths = lengths.cpu().to(torch.int64)  # 确保 lengths 在 CPU 上并且是 int64 类型
        outputs = model(inputs, lengths)

        # 计算所有时间步的损失，忽略填充部分的损失
        mask = (inputs[:, :, 0] != -1).float().unsqueeze(-1).to(device)
        loss = (criterion(outputs, target_sodium) * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()

        # print(f'Epoch [{epoch + 1}], Loss: {total_loss/ len(train_loader):.4f}')
        # print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])

    return total_loss / len(train_loader)


# 核心验证函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    loss_eval = 0
    with torch.no_grad():
        for inputs, lengths, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # target_sodium, target_sugar = targets.split(1, dim=2)  # 沿 dim=2 分割成两部分，每部分大小为 1
            target_sodium = targets
            lengths = lengths.cpu().to(torch.int64)  # 确保 lengths 在 CPU 上并且是 int64 类型
            outputs = model(inputs, lengths)
            mask = (inputs[:, :, 0] != -1).float().unsqueeze(-1).to(device)
            loss = (criterion(outputs, target_sodium) * mask).sum() / mask.sum()
            loss_eval += loss.item()
            avg_eval_loss = loss_eval / len(data_loader)
            # print(f'Eval Loss: {avg_eval_loss:.4f}')
    return avg_eval_loss
