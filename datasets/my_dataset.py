import os
import cv2
from torchvision import transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.class_names = sorted(os.listdir(root_dir))     # 读取根目录下的类别文件夹

        for label, class_name in enumerate(self.class_names):       # 给每个类别编号
            class_dir = os.path.join(root_dir, class_name)              # 拼接类别文件夹路径 eg: dataset/cats
            if not os.path.isdir(class_dir):        # 判断是不是文件夹
                continue
            for file_name in sorted(os.listdir(class_dir)):
                image_path = os.path.join(class_dir, file_name)
                if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):   # 判断是不是以这些结尾的图片
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((128, 128)),
        #     transforms.RandomHorizontalFlip(p=0.5),     # 有50%概率左右翻转图片
        #     transforms.RandomRotation(10),              # 随机旋转 ±10度
        #     transforms.ColorJitter(
        #         brightness=0.2,         # 亮度随机波动 ±20%
        #         contrast=0.2            # 对比度随机变化 ±20%
        #     ),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.5, 0.5, 0.5],
        #         std=[0.5, 0.5, 0.5]
        #     )
        # ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"图片读取失败:{image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, label