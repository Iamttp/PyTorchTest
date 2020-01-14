# PyTorch 原始方式实现softmax回归

import torch
import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np

# 如果不进行转换则返回的是PIL图片。
# transforms.ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片或者数据类型为np.uint8的NumPy数组转换为
# 尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor。
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

# 部分数据显示
# def get_fashion_mnist_labels(labels):
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
#
# def show_fashion_mnist(images, labels):
#     # 这里的_表示我们忽略（不使用）的变量
#     _, figs = plt.subplots(1, len(images), figsize=(12, 12))
#     for f, img, lbl in zip(figs, images, labels):
#         f.imshow(img.view((28, 28)).numpy())
#         f.set_title(lbl)
#         f.axes.get_xaxis().set_visible(False)
#         f.axes.get_yaxis().set_visible(False)
#     plt.show()
#
#
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取小批量
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)


# 测试： print(softmax(torch.tensor([[5., 1., -1.]])))
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)  # 同行元素求和
    return X_exp / partition  # 这里应用了广播机制


# view函数将每张原始图像改成长度为num_inputs的向量。
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 交叉熵损失函数
# 变量y_hat是2个样本在3个类别的预测概率，变量y是这2个样本的标签类别。通过使用gather函数，我们得到了2个样本的标签的预测概率。
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])
# print(y_hat.gather(1, y.view(-1, 1)))  # 取第一行的第0个，第二行的第2个
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 评价模型net在数据集data_iter上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            sgd(params, lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
