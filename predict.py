import torch
import cv2
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["cats", "dogs"]

# 创建模型结构，必须和训练时一致
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 2. 加载你训练好的模型
model.load_state_dict(
    torch.load("checkpoints/best_cat_dog_resnet18.pth", map_location=device)
)

model.eval()

# 3. 预测图片预处理，Resize 要和训练时一致
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 4. 图片路径
img_path = "data/test/cats/DM_20260518134950_001.png"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_tensor = transform(img)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

# 5. 推理
with torch.no_grad():
    outputs = model(input_tensor)
    _, pred = torch.max(outputs, dim=1)

print("预测结果:", class_names[pred.item()])