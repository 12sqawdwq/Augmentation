# AUTOAUGEMENT文章阅读笔记

### abstract

Data augmentation is an effective technique for improving the accuracy of modern image classifiers. However, cur-rent data augmentation implementations are manually de-signed. In this paper, we describe a simple procedure called AutoAugment to automatically search for improved data augmentation policies. In our implementation, we have de-signed a search space where a policy consists of many sub-policies, one of which is randomly chosen for each image in each mini-batch. A sub-policy consists of two opera-tions, each operation being an image processing function such as translation, rotation, or shearing, and the probabil-ities and magnitudes with which the functions are applied. We use a search algorithm to find the best policy such that the neural network yields the highest validation accuracy on a target dataset. Our method achieves state-of-the-art accuracy on CIFAR-10, CIFAR-100, SVHN, and ImageNet (without additional data). On ImageNet, we attain a Top-1 accuracy of 83.5% which is 0.4% better than the previous record of 83.1%. On CIFAR-10, we achieve an error rate of 1.5%, which is 0.6% better than the previous state-of-the-art. Augmentation policies we find are transferable between datasets. The policy learned on ImageNet transfers well to achieve significant improvements on other datasets, such as Oxford Flowers, Caltech-101, Oxford-IIT Pets, FGVC Air-craft, and Stanford Cars.

简单翻译便是，这篇文章讲述了一个实现auto algemetation的一种方法，通过规范化搜索空间

来实现脱离于传统的manully designed而是使用autoaugement去自动的search for合适的（improved）augementation policies

**自动搜索改进的数据增强策略**

- 数据增强是一种有效提高现代图像分类器准确性的技术。
- 当前的数据增强实现是手动设计的。
- 本文描述了一种名为AutoAugment的简单程序，用于自动搜索改进的数据增强策略。
- AutoAugment在CIFAR-10、CIFAR-100、SVHN和ImageNet（无额外数据）上实现了最先进的准确性。
- 在ImageNet上，达到了83.5%的Top-1准确率，比之前的记录高0.4%。
- 在CIFAR-10上，错误率为1.5%，比之前最好结果低0.6%。
- 所找到的增强策略可以在不同数据集之间迁移，并在其他数据集如Oxford Flowers、Caltech-101等上取得显著改进。

### **实现细节**

### **策略表示**

- 每个策略由多个子策略组成，每个子策略包含两个操作。
- 操作是图像处理函数，如平移、旋转或剪切，并附有应用的概率和幅度。
- 对于每个小批量中的每张图像，随机选择一个子策略进行应用。

### **搜索算法**

- 使用搜索算法找到最佳策略，使得神经网络在目标数据集上达到最高的验证准确性。
- 搜索空间包括不同的操作及其概率和幅度。
- 通过强化学习或进化算法优化策略。

### **结果评估**

- 在多个数据集上评估AutoAugment的效果，包括CIFAR-10、CIFAR-100、SVHN和ImageNet。
- 比较AutoAugment与其他方法的结果，展示其优越性。

**搜索空间的组成**

1. **策略（Policy）：**
    - 一个完整的策略由5个子策略（sub-policies）组成。
2. **子策略（Sub-policy）：**
    - 每个子策略包含两个图像处理操作（image operations），这些操作按顺序应用。
3. **操作（Operation）：**
- 每个操作与两个超参数相关联：
- **应用概率（Probability）：表示该操作被应用的概率。**
- **操作幅度（Magnitude）：表示该操作的强度或程度。**

```markdown
根据文件内容，测试集错误率是通过以下方式得出的：

- **数据增强模型对比**：文件中提到使用了两种不同的方法进行训练，并对比了它们在测试集上的错误率。具体来说：
  - **基线模型（Baseline models）**：使用标准的Inception预处理方法进行训练。
  - **AutoAugment-transfer模型**：使用在ImageNet上找到的策略进行数据增强后训练。

- **训练过程**：
  - 对于所有列出的数据集，训练了一个Inception v4模型1000个epoch，使用余弦学习率衰减和一个退火周期。
  - 学习率和权重衰减是根据验证集性能选择的。
  - 在确定了超参数后，将训练集和验证集合并再次训练。

- **图像尺寸**：所有图像大小设置为448x448像素。

因此，测试集错误率是对使用不同数据增强方法（即基线方法和AutoAugment-transfer方法）训练后的模型进行评估得出的。具体总结如下：

| 模型 | 数据增强方法 | 测试集错误率 |
| --- | ------------ | ------------ |
| Inception v4 | Baseline (Inception预处理) | 较高错误率 |
| Inception v4 | AutoAugment-transfer | 较低错误率 |

文件中明确指出，AutoAugment-transfer方法显著提高了FGVC数据集的泛化准确性，并且在Stanford Cars数据集上达到了最低的错误率。

综上所述，测试集错误率是通过对原有模型（基线模型）和使用AutoAugment-transfer进行数据增强后的模型进行训练并评估得出的。
```

代码部分：

```bash
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

```

## 代码解释

- `normalize` 定义了图像归一化的均值和标准差。
- `train_transform` 定义了训练数据的变换流程，包括随机裁剪、随机水平翻转、转换为张量和归一化。
- `test_transform` 定义了测试数据的变换流程，包括调整尺寸、中心裁剪、转换为张量和归一化。

```python

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# 训练
#定义训练数据的变换流程
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
# 测试
#定义测试数据的变换流程
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
```

定义一个自定义数据集类：

```python

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
```

这个类继承自 `torch.utils.data.Dataset`，并实现了 `__len__` 和 `__getitem__` 方法，用于加载和处理图像数据。

定义示例数据路径并显示一张图像：

```python
train_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\0a6ca56c-microUS_train_01.nii_layer_22.png"]
test_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\0a6ca56c-microUS_train_01.nii_layer_22.png"]
plt.imshow(Image.open(train_img_paths[0]))
train_dataset = CustomDataset(train_img_paths, transform=train_transform)
test_dataset = CustomDataset(test_img_paths, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

- `train_img_paths` 和 `test_img_paths` 定义了训练和测试数据的图像路径。
- 使用 `plt.imshow` 显示其中一张训练图像。
- `train_dataset` 和 `test_dataset` 创建了训练和测试数据集。
- `train_loader` 和 `test_loader` 创建了数据加载器，分别用于训练和测试数据。

### 加载图片的标准流程

1. **导入必要的库**：
    - 导入用于图像处理、数据加载和数据增强的库。例如，PIL（Pillow）、torchvision 和 matplotlib 等。
2. **定义图像路径**：
    - 指定训练和测试图像文件的路径。可以是一个列表，包含所有图像文件的路径。
3. **显示图像（可选）**：
    - 使用 `matplotlib` 等库显示一张或几张图像，以确保图像路径正确并且图像能够被正确加载。
4. **定义图像增强和预处理**：
    - 定义用于训练和测试数据的图像增强和预处理操作。例如，随机裁剪、随机翻转、归一化等。
5. **创建自定义数据集类**：
    - 创建一个继承自 `torch.utils.data.Dataset` 的自定义数据集类，实现 `__len__` 和 `__getitem__` 方法，以便加载和处理图像数据。
6. **创建数据集实例**：
    - 使用自定义数据集类创建训练和测试数据集实例，传入图像路径和预处理变换。
7. **创建数据加载器**：
    - 使用 `torch.utils.data.DataLoader` 创建数据加载器，指定批次大小和是否打乱数据。
8. **遍历数据加载器**：
    - 在训练或测试过程中，遍历数据加载器，获取批次数据进行训练或测试操作。

示例遍历训练和测试数据：

```python
# 示例：遍历训练数据
for images in train_loader:
    # 在这里进行训练操作
    pass

# 示例：遍历测试数据
for images in test_loader:
    # 在这里进行测试操作
    pass
```

这两段代码示例了如何遍历训练和测试数据。

示例：使用 GPU 和显示图片：

```python
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
```

- `device` 定义了使用 GPU 或 CPU。
- 遍历 `train_loader` 并显示一批训练图像。

自动增强中定义训练和测试数据的变换流程：

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
    transforms.RandomHorizontalFlip(hflip_prob),
    autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std)
])
```

这些变换流程包括随机裁剪、水平翻转、自动增强、转换为张量、转换数据类型和归一化。

再次定义示例数据路径并显示一张图像：

```python
train_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\1cde2c7a-microUS_train_17.nii_layer_16.png"]
test_img_paths = [r"E:\TOOL\datasets\deteset_for_calcification\tarin\images\1cde2c7a-microUS_train_17.nii_layer_16.png"]
plt.imshow(Image.open(train_img_paths[0]))
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
```