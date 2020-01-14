# ML & Pytorch

## 参考： python机器学习基础教程

---

`ML_1.py` 基本库的简单用法(numpy，SciPy，matplotlib，pandas)  

`ML_2.py` 基本ML使用(scikit-learn k邻近算法)

`ML_3.py` 基本ML使用(scikit-learn 线性回归)

`ML_4.py` 基本ML使用(scikit-learn 决策树)

---

`network.py` 使用numpy，SciPy搭建简单的三层神经网络

`regression.py` 使用pyTorch搭建简单的三层神经网络,用于回归

`classificiation.py` 使用pyTorch搭建简单的三层神经网络,用于分类

`cnn.py` 使用pyTorch搭建卷积神经网络,用于手写数字识别

---

## 《动手学深度学习》(PyTorch版)

`https://tangshusen.me/Dive-into-DL-PyTorch/#/`

---

`tensorTest.py` 预备知识，Tensor基本使用

`tensorTest2.py` PyTorch 原始方式实现线性回归

`tensorTest3.py` PyTorch 正常方式实现线性回归

`torchVisTest.py` PyTorch 原始方式实现softmax回归

`torchVisTest2.py` PyTorch 正常方式实现softmax回归

---

### 2. 预备知识

`torch.Tensor`是存储和变换数据的主要工具。Tensor和NumPy的多维数组非常类似。然而，Tensor提供GPU计算和自动求梯度等更多功能，这些使Tensor更加适合深度学习。

"tensor"这个单词一般可译作“张量”，张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。

### 3.1 线性回归

当模型和损失函数形式较为简单时，误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）。

然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）。

推荐文章： `https://blog.csdn.net/lilyth_lilyth/article/details/8973972`

和大多数深度学习模型一样，对于线性回归这样一种单层神经网络，它的基本要素包括模型、训练数据、损失函数和优化算法。

既可以用神经网络图表示线性回归，又可以用矢量计算表示该模型。

`torch.utils.data`模块提供了有关数据处理的工具，

`torch.nn`模块定义了大量神经网络的层，

`torch.nn.init`模块定义了各种初始化方法，

`torch.optim`模块提供了很多常用的优化算法。

### 3.4 softmax回归

虽然我们仍然可以使用回归模型来进行建模，并将预测值就近定点化到1、2、3等离散值之一，但这种连续值到离散值的转化通常会影响到分类质量。因此我们一般使用更加适合离散值输出的模型来解决分类问题。

`softmax`回归跟线性回归一样将输入特征与权重做线性叠加。与线性回归的一个主要不同在于，softmax回归的输出值个数等于标签里的类别数。

![](math.svg)

交叉熵适合衡量两个概率分布的差异。

`torchvision`包，它是服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。

`torchvision.datasets`: 一些加载数据的函数及常用的数据集接口；

`torchvision.models`: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；

`torchvision.transforms`: 常用的图片变换，例如裁剪、旋转等；

`torchvision.utils`: 其他的一些有用的方法。


### 3.8 多层感知机（multilayer perceptron，MLP）

多层感知机在输出层与输入层之间加入了一个或多个全连接隐藏层，并通过激活函数对隐藏层输出进行变换。

激活函数:

`ReLU`函数只保留正数元素，并将负数元素清零。

`sigmoid`函数。当输入接近0时，sigmoid函数接近线性变换。

`tanh`（双曲正切）函数可以将元素的值变换到-1和1之间

由于无法从训练误差估计泛化误差，一味地降低训练误差并不意味着泛化误差一定会降低。机器学习模型应关注降低泛化误差。

### `TODO` 3.12 权重衰减  3.13 丢弃法  3.14 正向传播、反向传播和计算图  3.15 数值稳定性和模型初始化

### 3.16 实战Kaggle比赛：房价预测

