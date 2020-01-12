# Regression 回归 f(x) = x^2 函数拟合

import matplotlib.pyplot as plt
import torch

# unsqueeze二维数据  squeeze压缩
# 因为torch只能处理二维的数据，所以我们用torch.unsqueeze给伪数据添加一个维度，dim表示添加在第几维
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.1 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# 训练网络之前我们需要先定义优化器和损失函数
# optimizer 是训练的工具
# Stochastic Gradient Descent (SGD) 随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
# 均方误差(MSE)
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

for t in range(1000):
    prediction = net(x)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 100 == 0:
        # plot and show learning process
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.show()
