# 预备知识

import torch

# .requires_grad设置为True，
# 它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

# x是直接创建的，所以它没有grad_fn, 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn。
# 像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None。
y = x + 2
print(y)
print(y.grad_fn)

# 梯度 grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。
# out = (1/4) * (3*(xi + 2)^2)求和1<=i<=4
z = y * y * 3
out = z.mean()
print(out)
out.backward()  # 等价于 out.backward(torch.tensor(1.))
print(x.grad)
