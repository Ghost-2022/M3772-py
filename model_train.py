import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train")
args = parser.parse_args()
mode = args.mode

# 定义转换操作，将图像转换为PyTorch需要的格式
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 定义数据集
train_dataset = datasets.ImageFolder(root=r'F:\Project_graduation\data\dataset\train', transform=transform)
val_dataset = datasets.ImageFolder(root=r'F:\Project_graduation\data\dataset\test', transform=transform)

# 数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, len(train_dataset.classes))

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CNNModel().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 训练模型
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=200):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 打印每个 batch 的进度和准确率
            if (i + 1) % 10 == 0:  # 每10个batch打印一次
                acc = 100 * correct / total  # 计算当前 epoch 到目前为止的准确率
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%')

        # 平均训练损失和准确率
        avg_train_loss = running_loss / (i + 1)
        train_acc = 100 * correct / total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # 验证模型
        avg_val_loss, val_acc = validate_model(model, val_loader, criterion)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # 每个 epoch 结束后打印损失和准确率
        print(
            f'Epoch {epoch + 1}/{num_epochs} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # 保存模型权重
    torch.save(model.state_dict(), r'F:\Project_graduation\data\models\model_pytorch_200.pth')
    return history


def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 平均验证损失和准确率
    avg_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    model.train()
    return avg_loss, val_acc


def plot_model_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Val Loss')
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='best')
    plt.show()


if mode == "train":
    history = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=200)
    plot_model_history(history)
