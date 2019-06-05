import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# -----------------------------------------------numpy
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n", format(x))

# -----------------------------------------------SciPy提供了使用多种数据结构创建稀疏矩阵的工具，以及将稠密矩阵转换为稀疏矩阵的工具。
eye = np.eye(4)
sparse_matrix = sparse.csr_matrix(eye)
print("sparse_matrix:\n", format(sparse_matrix))

# ----------------------------------------------%matplotlib notebook 或 %matplotlib inline
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker="x")
# jupyter notebook 可能不需要show
plt.show()

# ----------------------------------------------pandas
data = {'Name': ['ttp1', 'ttp2', 'ttp3'],
        'Age': [18, 19, 20]}
data_pandas = pd.DataFrame(data)
# 打印出美观的表格
display(data_pandas)