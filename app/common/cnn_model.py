import torch
from torchvision import transforms


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(64 * 12 * 12, 128)
        self.fc2 = torch.nn.Linear(128, 7)  # 这里假设你的数据集有5个类别

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义与训练时相同的转换操作
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 根据训练时的正则化方式设置
])

model = CNNModel().to(device)
model.load_state_dict(torch.load(r'../../static/models/model_pytorch_200.pth'))
model.eval()