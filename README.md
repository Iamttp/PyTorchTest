# ML & Pytorch

参考： python机器学习基础教程

`ML_1.py` 基本库的简单用法   

```python
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
```

`ML_2.py` 基本ML使用（scikit-learn）

```python
# 导入数据
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# 打乱数据并得到train,test data，大写X表示输入，二维数据。小写y表示输出，一维数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# 使用knn模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# 评估模型
knn.score(X_test,y_test)
```