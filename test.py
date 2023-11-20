import imp
import re
import jieba
from sklearn.base import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split

# 假设您已经有了预处理和分词后的数据
# 以及对应的标签

# 以下是一个假设的数据集
patents = ["专利文本1...", "Patent text 2...", ...]
labels = [0, 1, ...]

# 数据预处理（此处需要您根据实际情况填充）
def preprocess_data(text):
    # 分词、转换为整数序列等
    if re.search('[\u4e00-\u9fff]', text):  # 检测中文字符
        return " ".join(jieba.cut(text))
    else:  # 英文文本处理
        return text.lower()
    

# 分词和转换
processed_patents = preprocess_data(preprocess_data(patent) for patent in patents)

# 创建PyTorch数据集
X_train, X_test, y_train, y_test = train_test_split(processed_patents, labels, test_size=0.3)
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 构建 LSTM 网络模型
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        return x

# 实例化模型
model = LSTMClassifier(embedding_dim=100, hidden_dim=128, vocab_size=len(vocab))

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 评估模型
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs.squeeze() > 0.5).float()
    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f'Test Accuracy: {accuracy}')

# 应用模型进行预测（示例）
def predict_new_patent(model, new_patent):
    # 这里应该包括对新专利的预处理代码
    processed_patent = preprocess_data([new_patent])[0]
    patent_tensor = torch.tensor(processed_patent, dtype=torch.long)
    with torch.no_grad():
        output = model(patent_tensor)
        prediction = (output.squeeze() > 0.5).float()
    return prediction


new_patent = "新的专利描述文本..."
prediction = predict_new_patent(model, new_patent)
print(f'Prediction for new patent: {prediction.item()}')