import torch
from PIL import Image
from torchvision import transforms

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义与训练时相同的转换操作
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 根据训练时的正则化方式设置
])


# 加载模型
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


# 加载模型
model = CNNModel().to(device)
# model.load_state_dict(torch.load(r'F:\Project_graduation\data\models\model_pytorch_200.pth'))
model.load_state_dict(torch.load(r'static/models/model_pytorch_200.pth'))
model.eval()


# 定义预测函数
def predict_image(image_path):
    # 加载图片
    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
    image = transform(image).unsqueeze(0).to(device)  # 添加批次维度并转换为tensor

    # 进行预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # 选择概率最大的类别
        predicted_class = predicted.item()

    return predicted_class


# 测试用例
image_path = r"D:\work\Py\M3772\static\test_img\sad\微信图片_20250216130624.jpg"  # 替换为你要测试的图像路径
predicted_class = predict_image(image_path)

print(f'预测的类别是: {predicted_class}')
