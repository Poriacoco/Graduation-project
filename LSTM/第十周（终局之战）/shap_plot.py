import shap
import torch
import numpy as np
from model_sugar import LSTMModel
from data_address import load_data
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载训练好的模型
def load_trained_model():
    input_size = 20  # 根据实际输入特征维度修改
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(f"models/sugar_fold4.pt", map_location='cpu'))
    model.eval()
    return model

model=load_trained_model()
full_data=load_data('data/bio_dif_tar.xlsx')
X_train=full_data[0:10]
X_test=full_data[10:5]

# 数据预处理函数（需与训练时保持一致）
# def prepare_data(data):
#     processed = []
#     scaler_X = MinMaxScaler()
#     for df in data:
#         # 与train_s.py相同的预处理逻辑
#         time_feature = df['发酵周期/h'].values.reshape(-1, 1)
#         air_flow = np.array(df['风量L/h'], dtype=np.float32).reshape(-1, 1) / 10000
#         rotation_speed = np.array(df['转速r/min'], dtype=np.float32).reshape(-1, 1) / 1000
#         pressure_feature = np.array(df['罐压'], dtype=np.float32).reshape(-1, 1)
#
#         other_features = df.drop(
#             columns=['酸钠_diff', '残糖g/dl_diff', '发酵周期/h', '风量L/h', '转速r/min', '罐压'])
#
#         X = np.hstack([
#             time_feature / time_feature.max(),
#             air_flow,
#             rotation_speed,
#             pressure_feature,
#             scaler_X.fit_transform(other_features)
#         ])
#         processed.append(torch.FloatTensor(X))
#     return pad_sequence(processed, batch_first=True, padding_value=-1)


def analyze_packed_lstm_with_shap(model, X_train, X_test, feature_names, num_samples=10, background_samples=100):
        """
        对已训练好的使用packed padded序列的LSTM模型进行SHAP分析

        参数:
        model: 已训练好的LSTM模型
        X_train: 训练数据，用于背景分布
        X_test: 测试数据，用于解释
        feature_names: 特征名称列表
        num_samples: 要分析的测试样本数量
        background_samples: 用作背景的样本数量
        """
        # 确保模型处于评估模式
    model.eval()

        # 定义预测函数 - 这是SHAP将调用的函数
    def predict_fn(x_array):
            """将numpy数组转换为模型可接受的格式并返回预测结果"""
            # 转换为PyTorch张量
        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        batch_size = x_tensor.shape[0]

            # 计算每个序列的实际长度（非填充部分）
        lengths = []
        for i in range(batch_size):
                # 假设填充值为0，计算每个序列中非零行的数量
            length = (x_tensor[i].sum(axis=1) != 0).sum().item()
            lengths.append(max(1, length))  # 确保长度至少为1

            # 转换为张量并排序
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        lengths_tensor, sort_idx = lengths_tensor.sort(descending=True)
        x_sorted = x_tensor[sort_idx]

            # 打包序列
        packed_input = pack_padded_sequence(x_sorted, lengths_tensor, batch_first=True)

            # 进行预测
        with torch.no_grad():
            try:
                    # 尝试直接调用模型
                output = model(packed_input)
            except TypeError:
                    # 如果模型需要额外参数，可能需要调整
                output = model(packed_input, lengths_tensor)

            # 恢复原始顺序
        _, unsort_idx = sort_idx.sort(0)
        output = output[unsort_idx]

            # 返回numpy数组
        return output.cpu().numpy()

        # 选择背景数据
    if len(X_train) > background_samples:
        background_indices = np.random.choice(len(X_train), background_samples, replace=False)
        background_data = X_train[background_indices]
    else:
        background_data = X_train

        # 选择要解释的测试样本
    test_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    test_samples = X_test[test_indices]

    print(f"创建SHAP解释器，使用{len(background_data)}个背景样本...")
        # 创建KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, background_data)

    print(f"计算{len(test_samples)}个测试样本的SHAP值...")
        # 计算SHAP值
    shap_values = explainer.shap_values(test_samples)

        # 创建掩码来处理填充值
    masks = []
    for x in test_samples:
            # 对每个样本，创建一个掩码标记非填充位置
        mask = (x.sum(axis=1) != 0).astype(float)
            # 扩展掩码以匹配特征维度
        expanded_mask = np.repeat(mask[:, np.newaxis], x.shape[1], axis=1)
        masks.append(expanded_mask)

        # 应用掩码到SHAP值
    if isinstance(shap_values, list):
            # 多分类情况
        masked_shap_values = []
        for class_shap in shap_values:
            masked_class_shap = []
            for i, sv in enumerate(class_shap):
                masked_class_shap.append(sv * masks[i])
            masked_shap_values.append(masked_class_shap)
    else:
            # 二分类或回归情况
        masked_shap_values = []
        for i, sv in enumerate(shap_values):
            masked_shap_values.append(sv * masks[i])

        # 可视化结果
    print("生成SHAP摘要图...")
    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
            # 多分类情况
        shap.summary_plot(masked_shap_values[0], test_samples, feature_names=feature_names)
    else:
            # 二分类或回归情况
        shap.summary_plot(masked_shap_values, test_samples, feature_names=feature_names)

        # 为单个样本创建详细的可视化
    for i in range(min(3, len(test_samples))):
        print(f"生成样本 {i} 的详细SHAP分析...")
        visualize_sequence_shap(masked_shap_values, test_samples, feature_names,
                                    explainer.expected_value, i)

return explainer, shap_values, masked_shap_values

def visualize_sequence_shap(shap_values, data, feature_names, expected_value, sample_idx=0):
        """为时间序列数据创建详细的SHAP可视化"""
        # 获取样本的实际序列长度（非填充部分）
    seq_len = int((data[sample_idx].sum(axis=1) != 0).sum())

        # 提取非填充部分的数据
    sample_data = data[sample_idx][:seq_len]

        # 提取对应的SHAP值
    if isinstance(shap_values, list):
            # 多分类情况
        sample_shap = shap_values[0][sample_idx][:seq_len]
        ev = expected_value[0] if isinstance(expected_value, list) else expected_value
    else:
            # 二分类或回归情况
        sample_shap = shap_values[sample_idx][:seq_len]
        ev = expected_value

        # 1. 创建力图(Force Plot)
        plt.figure(figsize=(20, 6))
        shap.force_plot(ev, sample_shap, sample_data,
                        feature_names=feature_names, matplotlib=True, show=False)
        plt.title(f"样本 {sample_idx} 的SHAP力图")
        plt.tight_layout()
        plt.show()

        # 2. 创建时间序列热力图
        plt.figure(figsize=(15, 8))

        # 计算每个特征的平均重要性
        feature_importance = np.abs(sample_shap).mean(axis=0)
        sorted_idx = np.argsort(feature_importance)
        sorted_features = [feature_names[i] for i in sorted_idx]

        # 创建热力图
        plt.imshow(sample_shap[:, sorted_idx].T, aspect='auto', cmap='RdBu_r')
        plt.colorbar(label='SHAP值')
        plt.xlabel('时间步')
        plt.ylabel('特征')
        plt.title(f'样本 {sample_idx} 的时间序列SHAP值热力图')
        plt.yticks(np.arange(len(feature_names)), sorted_features)
        plt.tight_layout()
        plt.show()

        # 3. 为每个特征创建时间序列线图
        n_features = len(feature_names)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        plt.figure(figsize=(15, n_rows * 3))
        for i, feature_idx in enumerate(sorted_idx):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.plot(sample_shap[:, feature_idx], 'b-', label='SHAP值')
            plt.plot(sample_data[:, feature_idx], 'r--', label='特征值')
            plt.title(f'特征: {feature_names[feature_idx]}')
            plt.xlabel('时间步')
            plt.legend()

        plt.tight_layout()
        plt.show()


    # 特征名称（根据实际特征顺序）
feature_names = [
        '发酵周期/h',
        '风量L/h',
        '转速r/min',
        '罐压',
        *full_data[0].columns.drop([
            '酸钠_diff', '残糖g/dl_diff', '发酵周期/h',
            '风量L/h', '转速r/min', '罐压'])
    ]

# 进行SHAP分析
explainer, shap_values, masked_shap_values = analyze_packed_lstm_with_shap(
    model,
    X_train,
    X_test,
    feature_names,
    num_samples=3,  # 分析10个测试样本
    background_samples=30  # 使用100个背景样本
)

# 保存结果
np.save('shap_values.npy', shap_values)