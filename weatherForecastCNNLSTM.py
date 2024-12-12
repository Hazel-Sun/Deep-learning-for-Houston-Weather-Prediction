import pandas as pd  # 用于数据读取和处理的库
import numpy as np  # 进行矩阵运算的库
import matplotlib.pyplot as plt  # 数据可视化的绘图库
import os
import json
import torch  # 深度学习框架PyTorch
import torch.nn as nn  # PyTorch的神经网络模块
import torch.optim as optim  # PyTorch的优化器模块
import warnings  # 用于管理警告信息
warnings.filterwarnings('ignore')  # 忽略一些可以忽略的警告

# 设置随机种子
import random
torch.backends.cudnn.deterministic = True  # 将CuDNN随机性设置为确定模式
torch.backends.cudnn.benchmark = False  # 禁用CuDNN自动搜索最优算法功能
torch.manual_seed(42)  # 设置PyTorch随机种子
np.random.seed(42)  # 设置Numpy随机种子
random.seed(42)  # 设置Python随机种子

# 加载数据集并计算平均温度
data_2019 = pd.read_csv('HoustonWeather/Houston,TX 2019-01-01 to 2019-12-31.csv')
data_2020 = pd.read_csv('HoustonWeather/Houston,TX 2020-01-01 to 2020-12-31.csv')
data_2021 = pd.read_csv('HoustonWeather/Houston,TX 2021-01-01 to 2021-12-31.csv')
data_2022 = pd.read_csv('HoustonWeather/Houston,TX 2022-01-01 to 2022-12-31.csv')
data_2023 = pd.read_csv('HoustonWeather/Houston,TX 2023-01-01 to 2023-12-31.csv')

train_df = pd.concat([data_2019, data_2020, data_2021, data_2022, data_2023], ignore_index=True)

# 选择输入特征和目标变量
features = ['tempmax', 'tempmin']
target = 'temp'

X = train_df[features].values  # 输入特征
y = train_df[target].values  # 目标变量

# 数据归一化
from sklearn.preprocessing import MinMaxScaler

scaler_X = MinMaxScaler()  # 特征归一化
scaler_y = MinMaxScaler()  # 目标归一化

X = scaler_X.fit_transform(X)  # 归一化输入特征
y = scaler_y.fit_transform(y.reshape(-1, 1))  # 归一化目标变量

# 时间序列数据拆分
def split_data(data_X, data_y, time_step):
    """
    data_X: 输入特征
    data_y: 输出目标
    time_step: 时间步长
    """
    X, y = [], []
    for i in range(len(data_X) - time_step):
        X.append(data_X[i:i + time_step])  # 过去time_step天的特征
        y.append(data_y[i + time_step])  # 第time_step+1天的目标
    return np.array(X), np.array(y)

dataX, datay = split_data(X, y, time_step=32)
print(f"dataX.shape: {dataX.shape}, datay.shape: {datay.shape}")

# 数据集划分
def train_test_split(dataX, datay, shuffle=True, percentage=0.8):
    """
    将数据集划分为训练集和测试集
    shuffle：是否打乱数据顺序
    percentage：训练集占比
    """
    if shuffle:
        random_num = [index for index in range(len(dataX))]
        np.random.shuffle(random_num)  # 打乱数据
        dataX = dataX[random_num]
        datay = datay[random_num]
    split_num = int(len(dataX) * percentage)  # 按比例划分
    train_X = dataX[:split_num]
    train_y = datay[:split_num]
    test_X = dataX[split_num:]
    test_y = datay[split_num:]
    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = train_test_split(dataX, datay, shuffle=False, percentage=0.8)

class CNN_LSTM(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size):
        """
        初始化模型
        conv_input：卷积层输入通道数
        input_size：LSTM输入特征数
        hidden_size：LSTM隐藏层神经元数
        num_layers：LSTM层数
        output_size：全连接层输出特征数
        """
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1D卷积层
        self.conv = nn.Conv1d(in_channels=conv_input, out_channels=input_size, kernel_size=1)

        # LSTM层
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播
        x：输入数据
        """
        # 调整输入形状以适配 Conv1d
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, features) -> (batch_size, features, sequence_length)

        # 卷积操作
        x = self.conv(x)

        # 调整形状以适配 LSTM
        x = x.permute(0, 2, 1)  # (batch_size, features, sequence_length) -> (batch_size, sequence_length, features)

        # 初始化隐藏状态和记忆状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 创建模型
input_size = 12  # 输入特征数量
conv_input = len(features)  # 卷积层输入通道数
hidden_size = 64  # LSTM隐藏层维度
num_layers = 5  # LSTM层数
output_size = 1  # 预测目标（mean_temp）

model = CNN_LSTM(conv_input, input_size, hidden_size, num_layers, output_size)

# 模型训练
num_epochs = 500
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Adam优化器
criterion = nn.MSELoss()  # 均方误差损失函数

for epoch in range(num_epochs):
    model.train()
    num_batches = len(train_X) // batch_size
    for i in range(num_batches):
        batch_X = train_X[i * batch_size:(i + 1) * batch_size]
        batch_Y = train_y[i * batch_size:(i + 1) * batch_size]

        batch_X = torch.Tensor(batch_X)
        batch_Y = torch.Tensor(batch_Y)

        optimizer.zero_grad()
        output = model(batch_X)
        train_loss = criterion(output, batch_Y)
        train_loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            output = model(torch.Tensor(test_X))
            test_loss = criterion(output, torch.Tensor(test_y))
        print(f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}")

# 计算 R²
def r2_scoree(pred_y, true_y):
    ss_res = np.sum((true_y - pred_y) ** 2)
    ss_tot = np.sum((true_y - np.mean(true_y)) ** 2)
    return 1 - (ss_res / ss_tot)

# 计算 RMSE
def rmse(pred_y, true_y):
    return np.sqrt(np.mean((pred_y - true_y) ** 2))

# 计算 MAE
def mae(pred_y, true_y):
    return np.mean(np.abs(pred_y - true_y))

# 预测与反归一化（仅测试集）
test_pred = model(torch.Tensor(test_X)).detach().numpy()
test_pred = scaler_y.inverse_transform(test_pred)  # 反归一化测试集预测值
true_test_y = scaler_y.inverse_transform(test_y)  # 反归一化测试集真实值

# 计算测试集上的误差指标
r2_value = r2_scoree(test_pred, true_test_y)
rmse_value = rmse(test_pred, true_test_y)
mae_value = mae(test_pred, true_test_y)

print(f"MAE on Test Set: {mae_value}")
print(f"RMSE on Test Set: {rmse_value}")
print(f"R² on Test Set: {r2_value}")

# 可视化指标对比
plt.figure(figsize=(8, 6))
metrics = ['MAE', 'RMSE', 'R²']
values = [mae_value, rmse_value, r2_value]
bars = plt.bar(metrics, values, color=['blue', 'green', 'red'])
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', color='black') 

# plt.bar(metrics, values)
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title("Prediction Metrics with LSTM_CNN")

# Ensure the results folder exists
if not os.path.exists('result'):
    os.makedirs('result')

# Save the metrics plot
plt.savefig('result/prediction_metrics_LSTM_CNN.png')
# plt.show()

# 可视化测试集结果
plt.figure(figsize=(12, 6))
plt.plot(test_pred, label="Predicted Temperature with LSTM_CNN", marker="o", markersize=1)
plt.plot(true_test_y, label="Actual Temperature", marker="x", markersize=1)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title("Temperature Prediction with LSTM_CNN")
plt.legend()

# Save the temperature prediction plot
plt.savefig('result/temperature_prediction_LSTM_CNN.png')
plt.show()

# Output metrics in json
metrics_dict = {
    "Model": "CNN_LSTM",
    "MAE": mae_value,
    "RMSE": rmse_value,
    "R²": r2_value

}

with open('result/error_metrics_CNN_LSTM.json', 'w') as json_file:
    json.dump(metrics_dict, json_file)

print("json has been outputed")
