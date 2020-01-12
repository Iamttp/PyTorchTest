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

# ---------------------------------------------------------- 使用决策树，不设置max_depth在训练集的精度为100%
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))
