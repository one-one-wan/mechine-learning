import torch

# 给定数据集
data = torch.tensor([
    [1.0, 9.0],
    [1.1, 10.5],
    [2.0, 18.0],
    [3.0, 28.0],
    [3.2, 30.0],
    [4.0, 37.0],
    [5.0, 48.0],
    [1.2, 10.0]
], dtype=torch.float32)

X = data[:, 0:1]  # 特征值 x
y = data[:, 1]    # 目标值 y
# 用切片的方法分出输入值x和输出值y

# 在 X 前添加一列全为1的列，表示截距项
X_b = torch.cat([torch.ones((X.shape[0], 1)), X], dim=1)

### 标准方程法（闭式解）
theta_closed = torch.inverse(X_b.T @ X_b) @ X_b.T @ y
print("标准方程法得到的参数：", theta_closed)

# 使用标准方程法预测 x=3.5 和 x=4
X_test = torch.tensor([[1, 3.5], [1, 4]], dtype=torch.float32)
y_pred_closed = X_test @ theta_closed
print("标准方程法预测值（x=3.5和x=4）：", y_pred_closed)

### 梯度下降法
theta = torch.zeros((2, 1), requires_grad=True)  # 初始化参数为0
optimizer = torch.optim.SGD([theta], lr=0.01)  # 学习率0.01
n_epochs = 2000

# 重塑 y 以匹配矩阵维度
y = y.view(-1, 1)

for epoch in range(n_epochs):
    # 计算预测值和损失
    y_pred = X_b @ theta
    loss = torch.mean((y_pred - y) ** 2)

    # 反向传播并更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 梯度下降法得到的参数
print("梯度下降法得到的参数：", theta)

# 使用梯度下降法预测 x=3.5 和 x=4
y_pred_gd = X_test @ theta
print("梯度下降法预测值（x=3.5和x=4）：", y_pred_gd)

# 比较预测误差
error = torch.abs(y_pred_closed - y_pred_gd)
print("两种方法的预测误差：", error)
