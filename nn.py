import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#超参数
hidden_size = 1024
num_classes = 3  # 3个专利类别
learning_rate = 0.001
batch_size = 1280
num_epochs = 50

# 读取数据

class PatentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# feature 和 labels 是预处理后的特征和标签
features = sparse.load_npz("data/patent_verified_tfidf_matrix.npz")
df = pandas.read_csv("data/patent_verified.csv")
labels = df["y"].values.tolist()
input_size = features.shape[1]  # 特征向量的大小

X_train, X_test, y_train, y_test = train_test_split(features.toarray(), labels, test_size=0.3)
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.int)

dataset = PatentDataset(X_train_tensor, y_train_tensor)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# 定义模型

class PatentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PatentClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.softmax(out)
        return out


# 实例化模型

model = PatentClassifier(input_size, hidden_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
size = len(data_loader.dataset)
# 训练模型
for epoch in range(num_epochs):
    for batch, (inputs, targets) in enumerate(data_loader):
        # 前向传播
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.long())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{input_size:>5d}]")
            

X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.int)

model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor)
    predicted_classes = predicted.argmax(dim=1)
    # 将预测转换为整数类型
    predicted_classes = predicted_classes.int()
    print(y_test_tensor, predicted_classes)
    accuracy = accuracy_score(y_test_tensor, predicted_classes)
    print(f'Test Accuracy: {accuracy}')