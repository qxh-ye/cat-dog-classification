from datasets.my_dataset import MyDataset
from models.resnet18_model import create_model
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.engine import evaluate

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),     # 有50%概率左右翻转图片
        transforms.RandomRotation(10),              # 随机旋转 ±10度
        transforms.ColorJitter(
            brightness=0.2,         # 亮度随机波动 ±20%
            contrast=0.2            # 对比度随机变化 ±20%
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
train_dataset = MyDataset("data/train", transform=train_transform)
test_dataset = MyDataset("data/test", transform=test_transform)
val_dataset = MyDataset("data/val", transform=val_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(2)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.5
)


num_epochs = 5
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    val_acc = evaluate(model, val_loader, device)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_cat_dog_resnet18.pth")
        print("保存最佳模型")
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch[{epoch + 1}]/[{num_epochs}] "
        f"Loss:{avg_loss:.4f}   "
        f"Val Acc:{val_acc:.4f}   "
        f"Best Acc:{best_acc:.4f}   "
        f"LR: {current_lr}"
    )

    scheduler.step()

test_acc = evaluate(model, test_loader, device)
print("最终 Test Acc：", test_acc)