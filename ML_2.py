# ----------------------------------------------------------初识数据
from sklearn.datasets import load_iris

iris_dataset = load_iris()
# iris 鸢尾花  dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
# DESCR :简要说明， target_names :要预测的花的品种， feature_names : 每一特征进行说明
# data : 每一行代表一朵花，四个测量数据 target : 品种
print("初识数据\n数据数：", len(iris_dataset['target']))
print(iris_dataset.keys(), "\n")
print(iris_dataset['feature_names'])
print(iris_dataset['data'][45:55])
print(iris_dataset['target'][45:55])

# ----------------------------------------------------------训练数据和测试数据
from sklearn.model_selection import train_test_split

# 打乱数据，大写X表示输入，二维数据。小写y表示输出，一维数据
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
print("\n训练数据和测试数据\n", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ----------------------------------------------------------观察数据
# import pandas as pd
# from pandas.plotting import scatter_matrix
# import mglearn
#
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
#                      marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# ----------------------------------------------------------k邻近算法
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("\nk邻近算法\n注意格式为二维X_new.shape", X_new.shape)
prediction = knn.predict(X_new)
print("prediction：", iris_dataset['target_names'][prediction])

# ---------------------------------------------------------评估模型
y_pred = knn.predict(X_test)
print("\n评估模型\nscore：", np.mean(y_pred == y_test))
