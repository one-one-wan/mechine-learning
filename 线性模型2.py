import torch

# 数据
data = torch.tensor([
    [1, 2, 3, 20],
    [2, 3, 3, 25],
    [3, 2, 2, 21],
    [4, 2, 3, 28],
    [2, 3, 2, 22],
    [1, 2, 4, 23],
    [3, 3, 2, 25],
    [4, 4, 2, 29],
    [5, 5, 4, 43]
], dtype=torch.float32)

X = data[:, :3]  # 特征矩阵
y = data[:, 3]   # 目标值

# 在 X 前添加一列全为1的列，表示截距项
X_b = torch.cat([torch.ones((X.shape[0], 1)), X], dim=1)
# 用shape[0]求出行数，将形状为(9,1)的1矩阵与X列连结

# 标准方程法
theta_closed = torch.inverse(X_b.T @ X_b) @ X_b.T @ y
print("标准方程法得到的参数：", theta_closed)

# 预测 x1 = 3, x2 = 3, x3 = 3 时的 y 值
X_new = torch.tensor([1, 3, 3, 3], dtype=torch.float32)
y_pred_closed = X_new @ theta_closed
print("标准方程法预测值：", y_pred_closed.item())

# 使用梯度下降法
theta = torch.zeros(X_b.shape[1], requires_grad=True)  # 初始化参数为0
optimizer = torch.optim.SGD([theta], lr=0.001)  # 学习率0.001
n_epochs = 5000

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

# 预测 x1 = 3, x2 = 3, x3 = 3 时的 y 值
y_pred_gd = X_new @ theta
print("梯度下降法预测值：", y_pred_gd.item())