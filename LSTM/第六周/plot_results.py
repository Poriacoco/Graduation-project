import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def plot_results(true_values, predicted_values, attribute_name, style='modern',name='record'):
    """
    绘制真实值和预测值的对比图并保存
    true_values: 真实值数组
    predicted_values: 预测值数组
    attribute_name: 属性名称
    style: 可选风格 'modern', 'minimal', 'scientific', 'dark'
    """
    # 属性名称映射
    attribute_map = {
        '残糖g/dl_next': 'Residual Sugar',
        '酸钠_next':' Sodium'
    }

    # 获取英文属性名，如果没有映射则保持原样
    display_name = attribute_map.get(attribute_name, attribute_name)

    # 设置绘图风格
    if style == 'modern':
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ('#3498db', '#e74c3c')  # 蓝色和红色
        alpha = 0.8
        bg_color = '#f9f9f9'
    elif style == 'minimal':
        plt.style.use('seaborn-v0_8-white')
        colors = ('#2c3e50', '#e67e22')  # 深蓝和橙色
        alpha = 0.9
        bg_color = 'white'
    elif style == 'scientific':
        plt.style.use('ggplot')
        colors = ('#2980b9', '#c0392b')  # 蓝色和深红色
        alpha = 0.7
        bg_color = '#f5f5f5'
    elif style == 'dark':
        plt.style.use('dark_background')
        colors = ('#3498db', '#e74c3c')  # 亮蓝和亮红
        alpha = 0.8
        bg_color = '#2c3e50'
    else:
        plt.style.use('default')
        colors = ('blue', 'red')
        alpha = 0.7
        bg_color = 'white'

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(bg_color)

    # 计算误差统计
    errors = np.array(true_values) - np.array(predicted_values)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    r2 = r2_score(true_values, predicted_values)

    # 绘制真实值和预测值
    x = np.arange(len(true_values))
    ax.plot(x, true_values, 'o-', label=f'True values', color=colors[0],
            alpha=alpha, markersize=6, linewidth=1.5)
    ax.plot(x, predicted_values, 'x--', label=f'Predicted values', color=colors[1],
            alpha=alpha, markersize=6, linewidth=1.5)

    # 添加误差区域
    ax.fill_between(x, true_values, predicted_values, color=colors[1], alpha=0.15)

    # 设置坐标轴和网格
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel(display_name, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

    # 设置刻度字体
    ax.tick_params(axis='both', labelsize=10)

    # 添加标题和子标题
    ax.set_title(f'Comparison of Target vs Prediction: {display_name}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.text(0.5, -0.15, f'MAE: {mae:.4f} | RMSE: {rmse:.4f}| R2:{r2:.4f}',
            horizontalalignment='center', fontsize=11, transform=ax.transAxes)

    # 美化图例
    legend = ax.legend(loc='best', frameon=True, fontsize=10)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('lightgray')

    # 添加水印
    fig.text(0.99, 0.01, 'Created with Python', fontsize=8,
             color='gray', ha='right', va='bottom', alpha=0.5)

    # 调整布局
    plt.tight_layout()
    # 保存高质量图片
    plt.savefig(f'{name}{display_name}.png', dpi=300, bbox_inches='tight')
    # 返回图形对象，以便进一步自定义
    return fig, ax

def loss_plot(losses):
    plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 在代码开头添加以下字体设置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 尝试多个中文字体
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.savefig('training_loss.png')
    plt.close()