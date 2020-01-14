import torch
import torch.nn as nn
import pandas as pd

train_data = pd.read_csv('../kaggle_house/train.csv')
test_data = pd.read_csv('../kaggle_house/test.csv')

# 查看前4个样本的前4个特征、后2个特征和标签（SalePrice）
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 标准化（standardization） 1:-1 不用Id这个特征
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散数值转成指示特征。 dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape)  # (2919, 331) 特征数从79增加到了331。

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

# 训练模型
loss = torch.nn.MSELoss()


def get_net(feature_num):
    # net = nn.Linear(feature_num, 1)
    # for param in net.parameters():
    #     nn.init.normal_(param, mean=0, std=0.01)
    # return net
    net = nn.Sequential(
        nn.Linear(feature_num, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    return net


def train(net, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size):
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train(net, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('../kaggle_house/submission.csv', index=False)


num_epochs, lr, weight_decay, batch_size = 200, 5, 0, 64
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
