from timm.data.auto_augment import AugmentOp
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np, torch, torchvision.utils as vutils
from torchvision.transforms import autoaugment, transforms
from ultralytics.utils.plotting import plot_images

# 一般的图像增强baseline,imageNet数据集常用的图像增强方法
# 1. 随机裁剪 2. 随机翻转 3. 随机旋转
# 4. 随机缩放 5. 随机亮度 6. 随机对比度
# 7. 随机饱和度 8. 随机色调 9. 随机噪声 10. 随机模糊
# 11. 随机旋转 12. 随机仿射变换 13. 随机透视变换 14. 随机擦除

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# 训练
train_transform = transforms.Compose([
    # 这里的scale指的是面积，ratio是宽高比
    # 具体实现每次先随机确定scale和ratio，可以生成w和h，然后随机确定裁剪位置进行crop
    # 最后是resize到target size
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
# 测试
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# 示例数据集类
class CustomDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 示例数据路径
train_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\0a6ca56c-microUS_train_01.nii_layer_22.png"]
test_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\0a6ca56c-microUS_train_01.nii_layer_22.png"]
plt.imshow(Image.open(train_img_paths[0]))
# 创建数据集和数据加载器
train_dataset = CustomDataset(train_img_paths, transform=train_transform)
test_dataset = CustomDataset(test_img_paths, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 示例：遍历训练数据
for images in train_loader:
    # 在这里进行训练操作
    pass

# 示例：遍历测试数据
for images in test_loader:
    # 在这里进行测试操作
    pass

# 示例：使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例：显示图片
for images in train_loader:
    plt.figure(figsize=(16, 16))
    plt.axis("on")
    plt.title("Training Images")
    plt.xlabel("Batch")
    plt.ylabel("Images")
    plt.imshow(np.transpose(vutils.make_grid(images.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    plt.pause(0.001)
    break

#这里使用的使torchvision.transforms.autoaugment.AutoAugmentPolicy.IMAGENET
#属于ImageNet数据集的自动增强策略
#论文来自于：https://arxiv.org/abs/1805.09501
# AutoAugment: Learning Augmentation Policies from Data

# crop_size：用于指定裁剪的尺寸。
# interpolation：用于指定插值方法。
# hflip_prob：用于指定水平翻转的概率。
# aa_policy：用于指定自动增强策略。
# mean 和 std：用于指定图像归一化的均值和标准差。

crop_size = 224
interpolation = transforms.InterpolationMode.BILINEAR
hflip_prob = 0.5
aa_policy = autoaugment.AutoAugmentPolicy.IMAGENET
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# transforms.RandomResizedCrop(crop_size, interpolation=interpolation)：随机裁剪图像到指定的尺寸crop_size，并使用指定的插值方法interpolation进行缩放。
# transforms.RandomHorizontalFlip(hflip_prob)：以概率hflip_prob随机水平翻转图像。
# autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation)：使用自动增强策略aa_policy（这里是ImageNet的策略）对图像进行增强，并使用指定的插值方法interpolation。
# transforms.PILToTensor()：将PIL图像转换为PyTorch张量。
# transforms.ConvertImageDtype(torch.float)：将图像数据类型转换为浮点型。
# transforms.Normalize(mean=mean, std=std)：使用指定的均值mean和标准差std对图像进行归一化。

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
    transforms.RandomHorizontalFlip(hflip_prob),
    # 这里policy属于torchvision.transforms.autoaugment.AutoAugmentPolicy，
    # 对于ImageNet就是 AutoAugmentPolicy.IMAGENET
    # 此时aa_policy = autoaugment.AutoAugmentPolicy('imagenet')
    autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std)
    ])

# 测试
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std)
])

# 示例数据路径
train_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\1cde2c7a-microUS_train_17.nii_layer_16.png"]
test_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\1cde2c7a-microUS_train_17.nii_layer_16.png"]
plt.imshow(Image.open(train_img_paths[0]))
# 创建数据集和数据加载器
train_dataset = CustomDataset(train_img_paths, transform=train_transform)
test_dataset = CustomDataset(test_img_paths, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 示例：遍历训练数据
for images in train_loader:
    # 在这里进行训练操作
    pass

# 示例：遍历测试数据
for images in test_loader:
    # 在这里进行测试操作
    pass

# 示例：使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例：显示图片
for images in train_loader:
    plt.figure(figsize=(16, 16))
    plt.axis("on")
    plt.title("Training Images")
    plt.xlabel("Batch")
    plt.ylabel("Images")
    plt.imshow(np.transpose(vutils.make_grid(images.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    plt.pause(0.001)
    break

#此处的代码为使用all_policy_use_op进行数据增强效果实验
# img_path = r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\0a6ca56c-microUS_train_01.nii_layer_22.png"
# mean = (0.485, 0.456, 0.406)
# X = Image.open(img_path).convert('RGB')
# img_size_min = min(X.size)
# plt.imshow(X)
# plt.show()
#
# all_policy_use_op = [
#     ['AutoContrast', 1, 10], ['Equalize', 1, 10], ['Invert', 1, 10], ['Rotate', 1, 10], ['Posterize', 1, 10],
#     ['PosterizeIncreasing', 1, 10], ['PosterizeOriginal', 1, 10], ['Solarize', 1, 10], ['SolarizeIncreasing', 1, 10],
#     ['SolarizeAdd', 1, 10], ['Color', 1, 10], ['ColorIncreasing', 1, 10], ['Contrast', 1, 10],
#     ['ContrastIncreasing', 1, 10], ['Brightness', 1, 10], ['BrightnessIncreasing', 1, 10], ['Sharpness', 1, 10],
#     ['SharpnessIncreasing', 1, 10], ['ShearX', 1, 10], ['ShearY', 1, 10], ['TranslateX', 1, 10], ['TranslateY', 1, 10],
#     ['TranslateXRel', 1, 10], ['TranslateYRel', 1, 10]
# ]
#
# for op_name, p, m in all_policy_use_op:
#     aug_op = AugmentOp(name=op_name, prob=p, magnitude=m,
#                        hparams={'translate_const': int(img_size_min * 0.45),
#                                 'img_mean': tuple([min(255, round(255 * x)) for x in mean])})
#     plt.imshow(aug_op(X))
#     plt.title(f'{op_name}_{str(p)}_{str(m)}')
#     plt.show()
