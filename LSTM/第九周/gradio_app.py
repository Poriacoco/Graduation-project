import gradio as gr
import torch
import numpy as np
from model_sugar import LSTMModel
from address_data import load_data
from sklearn.preprocessing import MinMaxScaler
import joblib


# 加载预训练的模型和归一化器
def load_resources():
    # 加载模型
    input_size = 20  # 必须与训练时的特征维度一致
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load("sugar_fold_best.pt", map_location='cpu'))
    model.eval()

    # 加载保存的归一化器（需要提前保存训练时的scaler）
    # 假设已经保存了scaler_X和scaler_Y2
    scaler_X = joblib.load('scaler_X.save')
    scaler_Y2 = joblib.load('scaler_Y2.save')

    return model, scaler_X, scaler_Y2


model, scaler_X, scaler_Y2 = load_resources()

# 特征顺序必须与训练时完全一致
feature_order = [
    '发酵周期/h', '风量L/h', '转速r/min', '罐压',
    '酶活', '酸钠', '残糖g/dl', '菌浓ml/50ml',
    '菌浓g/50ml', 'PH值', '溶氧', '温度',
    '酶活_差值', '酸钠_差值', '残糖g/dl_差值',
    '菌浓ml/50ml_差值', '菌浓g/50ml_差值',
    '溶氧_差值', '碱重kg_差值', '重量KG_差值'
]


def predict(*inputs):
    # 将输入转换为numpy数组
    input_array = np.array(inputs, dtype=np.float32).reshape(1, -1)

    # 数据预处理（与训练时一致）
    time_feature = input_array[:, 0].reshape(-1, 1) / 24.0  # 假设最大发酵周期为24小时
    air_flow = input_array[:, 1].reshape(-1, 1) / 10000
    rotation_speed = input_array[:, 2].reshape(-1, 1) / 1000
    pressure = input_array[:, 3].reshape(-1, 1)

    # 对其他特征进行归一化
    other_features = scaler_X.transform(input_array[:, 4:])

    # 组合所有特征
    processed = np.hstack([
        time_feature,
        air_flow,
        rotation_speed,
        pressure,
        other_features
    ])

    # 转换为tensor并添加序列维度
    tensor_input = torch.FloatTensor(processed).unsqueeze(0)
    lengths = torch.tensor([1])  # 序列长度为1

    # 进行预测
    with torch.no_grad():
        prediction = model(tensor_input, lengths)

    # 反归一化
    prediction = scaler_Y2.inverse_transform(prediction.numpy())

    return f"预测下一时刻残糖含量：{prediction[0][0]:.2f} g/dl"


# 创建输入组件
inputs = []
for feature in feature_order:
    if "风量" in feature:
        inputs.append(gr.Slider(0, 3000, step=100, label=feature))
    elif "转速" in feature:
        inputs.append(gr.Slider(0, 1000, step=50, label=feature))
    elif "温度" in feature:
        inputs.append(gr.Slider(30, 50, step=0.1, label=feature))
    elif "差值" in feature:
        inputs.append(gr.Number(label=feature))
    else:
        inputs.append(gr.Number(label=feature))

# 创建界面
interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="发酵过程残糖含量预测",
    description="输入当前发酵参数，预测下一时刻的残糖含量",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()