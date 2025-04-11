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
from sklearn.model_selection import KFold

from model_sodium import train_model ,evaluate_model,LSTMModel
from address_data import load_data
from plot_results import plot_results,loss_plot

def set_seed(seed=42, deterministic=True):
    """
    设置所有可能的随机种子，确保 PyTorch 实验可重复性
    参数:
        seed (int): 随机种子值
        deterministic (bool): 是否使用确定性算法
    """
    # Python 内置随机模块
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # CUDA 操作的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    # 设置 CUDA 后端的确定性选项
    if deterministic:
        # 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 设置环境变量
        os.environ['PYTHONHASHSEED'] = str(seed)
        # PyTorch 2.0+ 的额外设置
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        elif hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)

    print(f"随机种子已设置为: {seed}")
    if deterministic:
        print("已启用确定性模式")

#设置随机数种子
set_seed(717)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_validation(full_data, n_splits=5, num_epochs=100):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=717)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_data)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # 数据划分
        train_data = [full_data[i] for i in train_idx]
        val_data = [full_data[i] for i in val_idx]

        # 独立归一化处理
        scaler_X = MinMaxScaler()
        scaler_Y1 = MinMaxScaler()
        scaler_Y2 = MinMaxScaler()


        # 训练集处理
        X_train, y1_train,y2_train = [], [], []
        for df in train_data:
            # 处理发酵周期
            time_feature = df['发酵周期/h'].values.reshape(-1, 1)
            # 处理风量列
            air_flow = np.array(df['风量L/h'], dtype=np.float32).reshape(-1, 1) / 10000
            # 处理转速列
            rotation_speed = np.array(df['转速r/min'], dtype=np.float32).reshape(-1, 1) / 1000
            other_features = df.drop(columns=['酸钠_next', '残糖g/dl_next', '发酵周期/h','风量L/h','转速r/min'])

            # 归一化
            X = np.hstack([
                time_feature/time_feature.max(),
                air_flow,
                rotation_speed,
                scaler_X.fit_transform(other_features)
            ])

            y1 = scaler_Y1.fit_transform(df['酸钠_next'].values.reshape(-1, 1))
            y2 = scaler_Y2.fit_transform(df['残糖g/dl_next'].values.reshape(-1, 1))


            X_train.append(torch.FloatTensor(X))
            y1_train.append(torch.FloatTensor(y1))
            y2_train.append(torch.FloatTensor(y2))


        # 验证集处理（使用训练集的scaler）
        X_val, y1_val, y2_val= [], [], []
        for df in val_data:
            time_feature = df['发酵周期/h'].values.reshape(-1, 1)
            # 处理风量列
            air_flow = np.array(df['风量L/h'], dtype=np.float32).reshape(-1, 1) / 10000
            # 处理转速列
            rotation_speed = np.array(df['转速r/min'], dtype=np.float32).reshape(-1, 1) / 1000
            other_features = df.drop(columns=['酸钠_next', '残糖g/dl_next', '发酵周期/h', '风量L/h', '转速r/min'])

            X = np.hstack([
                time_feature / time_feature.max(),
                air_flow,
                rotation_speed,
                scaler_X.transform(other_features)
            ])

            y1 = scaler_Y1.transform(df['酸钠_next'].values.reshape(-1, 1))
            y2 = scaler_Y2.transform(df['残糖g/dl_next'].values.reshape(-1, 1))

            X_val.append(torch.FloatTensor(X))
            y1_val.append(torch.FloatTensor(y1))
            y2_val.append(torch.FloatTensor(y2))

        # 数据封装
        train_dataset = TensorDataset(
            pad_sequence(X_train, batch_first=True, padding_value=-1),
            torch.tensor([x.shape[0] for x in X_train]),
            pad_sequence(y1_train, batch_first=True, padding_value=-1)
        )
        val_dataset = TensorDataset(
            pad_sequence(X_val, batch_first=True, padding_value=-1),
            torch.tensor([x.shape[0] for x in X_val]),
            pad_sequence(y1_val, batch_first=True, padding_value=-1)
        )
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)




        # 架构超参数
        hidden_size = 128
        num_layers = 3
        output_size = 1  # 设置输出大小为1
        # 训练超参数
        num_epochs = 70
        best_loss = 99999
        best_eval_loss = 99999
        dropout_rate = 0.08

        # 模型初始化
        input_size = X_train[0].shape[1]
        model = LSTMModel(input_size,hidden_size,num_layers,output_size,dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=6e-4)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.85)
        criterion = nn.MSELoss()

        # 训练循环
        best_val_loss = float('inf')
        patience = 5  # 早停容忍轮次
        no_improve = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, optimizer, criterion, device,epoch)
            val_loss = evaluate_model(model, val_loader, criterion, device)
            scheduler.step()
            #记录历史数据（便于后续打印）
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 仅保存模型参数
                torch.save(model.state_dict(), f"best_fold{fold}.pt")
                no_improvement = 0
            else:
                no_improvement += 1

            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            #早停检查
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # 最终评估
        model.load_state_dict(torch.load(f"best_fold{fold}.pt"))
        final_val_loss = evaluate_model(model, val_loader, criterion, device)

        # 反归一化并计算指标
        all_preds, all_true = [], []
        with torch.no_grad():
            for inputs, lengths, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # target_sodium, target_sugar = targets.split(1, dim=2)  # 沿 dim=2 分割成两部分，每部分大小为 1
                target_sodium = targets
                outputs = model(inputs, lengths)

                all_preds.append(outputs.cpu().numpy())
                all_true.append(target_sodium.cpu().numpy())



        preds = np.concatenate(all_preds)
        true = np.concatenate(all_true)
        preds = scaler_Y1.inverse_transform(preds.reshape(-1, 1))
        true = scaler_Y1.inverse_transform(true.reshape(-1, 1))

        mae = mean_absolute_error(true, preds)
        r2 = r2_score(true, preds)

        results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'mae': mae,
            'r2': r2
        })

    # 输出综合结果
    print("\n=== Cross Validation Results ===")
    df_results = pd.DataFrame(results)
    print(df_results)

    print("\n=== Average Metrics ===")
    print(f"Average Val Loss: {df_results['final_val_loss'].mean():.4f} ± {df_results['final_val_loss'].std():.4f}")
    print(f"Average MAE: {df_results['mae'].mean():.2f} ± {df_results['mae'].std():.2f}")
    print(f"Average R²: {df_results['r2'].mean():.2f} ± {df_results['r2'].std():.2f}")

    return df_results



# 主程序
if __name__ == "__main__":
    # 加载完整数据
    full_data = load_data('bio_all_34.xlsx')

    # 执行交叉验证
    results = cross_validation(full_data, n_splits=11, num_epochs=80)

    # 可视化结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(results['fold'], results['mae'])
    plt.title('MAE per Fold')

    plt.subplot(1, 2, 2)
    plt.bar(results['fold'], results['r2'])
    plt.title('R² Score per Fold')
    plt.tight_layout()
    plt.show()