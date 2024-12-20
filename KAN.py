import pandas as pd
import numpy as np
import tensorflow as tf
from tfkan.layers import DenseKAN
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pygmt
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 参数，确保中文显示正常
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 读取数据集并导入水深控制点和检查点
region = "140/148/10/16"
data = pygmt.select(region=region, data="Training_dataset_all.txt")
data_pre = pygmt.select(region=region, data="Prediction_dataset_all.txt")

# 特征和目标变量的提取与处理，去掉经纬度等无关列
X = data.drop(data.columns[[ 0,1]], axis=1)
y = data.iloc[:, 2]
X_pre = data_pre.drop(data.columns[[ 0,1]], axis=1)
y_pre = data_pre.iloc[:, 2]

# 归一化数据
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
X_pre_scaled = scaler_X.fit_transform(X_pre)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
y_pre_scaled = scaler_y.fit_transform(y_pre.values.reshape(-1, 1))

# 划分训练集和测试集
X_train = X_scaled
X_test = X_pre_scaled
y_train = y_scaled
y_test = y_pre_scaled

# 定义 KAN 模型
class KAN_Model(models.Model):
    def __init__(self, input_dim, layer_units):
        super(KAN_Model, self).__init__()
        self.dense_kan1 = DenseKAN(layer_units)
        self.dense_kan2 = DenseKAN(1)  # 假设输出为1维

    def call(self, inputs):
        x = self.dense_kan1(inputs)
        x = self.dense_kan2(x)
        return x

# 超参数设置
input_dim = X_train.shape[1]  # 输入维度
layer_units = 512  # 隐藏层神经元数量

# 创建并构建 KAN 模型
kan_model = KAN_Model(input_dim, layer_units)
kan_model.build(input_shape=(None, input_dim))
kan_model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
kan_model.summary()

# 训练模型
history = kan_model.fit(X_train, y_train, epochs=100, batch_size=512, validation_split=0.3)

# 预测
y_pred_scaled = kan_model.predict(X_test)

# 反归一化预测值
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test)

# 评估模型
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 打印评估指标
print(f'平均绝对误差 (MAE): {mae}')
print(f'均方误差 (MSE): {mse}')
print(f'均方根误差 (RMSE): {rmse}')
print(f'R-平方 (R2): {r2}')


# 绘制训练集和验证集的损失值收敛图
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('训练集和验证集的损失值收敛图')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制真实值和预测值的对比曲线
plt.figure(figsize=(8, 6))
plt.plot(y_test, label='真实值', color='green', linewidth=2)
plt.plot(y_pred, label='预测值', color='red', linestyle='dashed', linewidth=2)
plt.xlabel('样本序号', fontsize=12)
plt.ylabel('目标值', fontsize=12)
plt.title('KAN 预测 vs 真实值', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 绘制真实值与预测值的散点图和拟合线
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='#8c564b', marker='o', label='预测值', alpha=0.7)
plt.plot(y_test, y_test, color='#d62728', linestyle='-', label='y=x', linewidth=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('真实值 vs 预测值')
plt.legend()
plt.grid(True, linestyle='-', alpha=0.7)
plt.tight_layout()
plt.show()

# 绘制误差直方分布图
plt.figure(figsize=(8, 6))
errors = y_test - y_pred
plt.hist(errors, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('误差', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('误差直方分布图', fontsize=14)
mean_error = np.mean(errors)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'平均误差 = {mean_error:.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# 保存真实值与预测值到文件
data_pre_columns = data_pre.iloc[:, :2]  # 读取 data_pre 的前两列
result_df = pd.DataFrame({
    '列1': data_pre_columns.iloc[:, 0],
    '列2': data_pre_columns.iloc[:, 1],
    '真实值': y_test.flatten(),
    '预测值': y_pred.flatten()
})

# 将 DataFrame 保存为 kan_all.txt 文件，使用制表符作为分隔符
result_df.to_csv('kan_cpu.txt', sep='\t', index=False)

import pygmt
region = "140/148/10/16"
data = pygmt.select(region=region, data="kan_cpu.txt")   # 预测结果
data = data.iloc[:, [0, 1, 3]] # 0,1为坐标，2为预测

fig = pygmt.Figure()
pygmt.surface(region="140/148/10/16", spacing="15s", outgrid="kan_cpu.grd", data=data)
fig.basemap(region="140/148/10/16", projection="M15c", frame="afg")
fig.coast(borders=["1/0.5p,black", "2/0.5p,red", "3/0.5p,blue"], land="gray")
fig.grdimage(
    grid="kan_cpu.grd",
    cmap="rainbow",
    shading=True,  # 添加光照效果
    frame=True,
)
fig.colorbar(frame=["a", "x+lElevation", "y+lm"])

fig.show(width="1000")
