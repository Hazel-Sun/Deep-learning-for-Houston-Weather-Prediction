import json
import matplotlib.pyplot as plt
import numpy as np

# 定义要读取的JSON文件路径
json_files = [
    'result/error_metrics_CNN_LSTM.json',
    'result/error_metrics_TCN.json',
    'result/error_metrics_GRU.json',
    'result/error_metrics_LSTM.json'
]

# 初始化数据存储
models = []
mae_values = []
# mse_values = []
rmse_values = []
r2_values = []

# 读取每个 JSON 文件的数据
for json_file in json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
        models.append(data['Model'])
        mae_values.append(data['MAE'])
        # mse_values.append(data['MSE'])
        rmse_values.append(data['RMSE'])
        r2_values.append(data['R²'])


# 定义柱状图宽度和位置
x = np.arange(3)  # MAE, RMSE, R²三个组的位置
bar_width = 0.2  # 柱状图宽度

# 创建图表
plt.figure(figsize=(12, 5))

# 每种模型用不同颜色的柱状图
colors = ['blue', 'green', 'orange', 'red']

# 绘制柱状图
for i, model in enumerate(models):
    bars = plt.bar(
        x + i * bar_width,  # 根据模型索引调整位置
        # [mae_values[i], mse_values[i], rmse_values[i]],  # 每种误差类型的值
        [mae_values[i], rmse_values[i],  r2_values[i]],  # 每种误差类型的值

        width=bar_width,
        label=model,
        color=colors[i]
    )
    # 在每个柱子上显示具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:.2f}', ha='center', va='bottom'
        )

# 添加标题和轴标签
plt.title('Error Metrics Comparison by Model', fontsize=16)
plt.xlabel('Error Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(x + bar_width * 1.5, ['MAE', 'RMSE', 'R²'])  # 设置 x 轴刻度为误差类型，居中显示

# 添加图例
plt.legend()

# 显示图表
plt.tight_layout()
plt.savefig('result/comparison_error_metrics.png')
plt.show()
