import shap
import torch
import numpy as np
from model_sugar import LSTMModel
from address_data import load_data
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载训练好的模型
def load_trained_model():
    input_size = 20  # 根据实际输入特征维度修改
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(f"sugar_fold_best.pt", map_location='cpu'))
    model.eval()
    return model


# 数据预处理函数（需与训练时保持一致）
def prepare_data(data):
    processed = []
    scaler_X = MinMaxScaler()
    for df in data:
        # 与train_s.py相同的预处理逻辑
        time_feature = df['发酵周期/h'].values.reshape(-1, 1)
        air_flow = np.array(df['风量L/h'], dtype=np.float32).reshape(-1, 1) / 10000
        rotation_speed = np.array(df['转速r/min'], dtype=np.float32).reshape(-1, 1) / 1000
        pressure_feature = np.array(df['罐压'], dtype=np.float32).reshape(-1, 1)

        other_features = df.drop(
            columns=['酸钠_next', '残糖g/dl_next', '发酵周期/h', '风量L/h', '转速r/min', '罐压'])

        X = np.hstack([
            time_feature / time_feature.max(),
            air_flow,
            rotation_speed,
            pressure_feature,
            scaler_X.fit_transform(other_features)
        ])
        processed.append(torch.FloatTensor(X))
    return pad_sequence(processed, batch_first=True, padding_value=-1)


# 包装模型前向传播
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 自动生成长度信息（假设所有样本长度相同）
        valid_lengths = (x[:, :, 0] != -1).sum(dim=1).cpu().to(torch.int64)
        outputs = self.model(x, valid_lengths)
        mask = (x[:, :, 0] != -1).float().unsqueeze(-1)
        return outputs * mask


# 主解释函数
def explain_model():
    # 加载数据和模型
    full_data = load_data('bio_add_features.xlsx')
    model = load_trained_model()
    wrapped_model = ModelWrapper(model)

    # 准备背景数据（取前100个样本）
    background_data = prepare_data(full_data[:30])

    # 选择要解释的样本（例如第一个样本）
    sample_idx = 2
    sample_data = prepare_data([full_data[sample_idx]])

    #创建解释器
    explainer = shap.DeepExplainer(
        wrapped_model,
        background_data,
        # feature_perturbation="interventional",
        # tolerance=0.05
    )
    #
    #
    # # 计算SHAP值
    # shap_values = explainer.shap_values(sample_data)
    # # 验证结果一致性
    # model_output = wrapped_model(sample_data).detach().numpy()
    # shap_sum = explainer.expected_value + shap_values[0].sum()

    # 初始化解释器
    # 计算 SHAP 值
    shap_values = explainer.shap_values(background_data)

    # 特征名称（根据实际特征顺序）
    feature_names = [
        '发酵周期/h',
        '风量L/h',
        '转速r/min',
        '罐压',
        *full_data[0].columns.drop([
            '酸钠_next', '残糖g/dl_next', '发酵周期/h',
            '风量L/h', '转速r/min', '罐压'
        ])
    ]




    #所有样本的shap值
    # mean_shap_values = shap_values[0].mean(axis=1)
    # # 绘制 SHAP 值的 summary plot (按平均值聚合)
    # plt.figure()
    # shap.summary_plot(mean_shap_values, background_data[:, 0, :], feature_names=feature_names, plot_type="dot")




    # 可视化第一个时间步的解释
    shap.initjs()
    # time_step = 0
    # print(f"特征重要性（时间步 {time_step + 1}）:")
    # shap.force_plot(
    #     explainer.expected_value[0][time_step],
    #     shap_values[0][0][time_step],
    #     feature_names=feature_names,
    #     matplotlib=True
    # )
    sample_idx = 0
    for time_step in range(shap_values[0].shape[1]):
        plt.figure()
        shap.force_plot(
            explainer.expected_value[time_step],
            shap_values[0][sample_idx][time_step],
            feature_names=feature_names,
            matplotlib=True
        )
        plt.title(f"样本 {sample_idx+1} - 时间步 {time_step+1}")
        plt.show()

if __name__ == "__main__":
    explain_model()