import os
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import autoaugment

# 定义图像转换操作
crop_size = 224
interpolation = transforms.InterpolationMode.BILINEAR
hflip_prob = 0.5
aa_policy = autoaugment.AutoAugmentPolicy.IMAGENET
mean = [0.485]
std = [0.5]

transform = transforms.Compose([
    transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
    transforms.RandomHorizontalFlip(hflip_prob),
    autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std)
])

# 定义不处理颜色的操作
no_color_transform = transforms.Compose([
    transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
    transforms.RandomHorizontalFlip(hflip_prob),
    autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

# 源文件夹和目标文件夹路径
source_image_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\images'
source_label_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\labels'
target_image_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\processed_images_aug_v2\images'
target_label_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\processed_images_aug_v2\labels'

# 创建目标文件夹（如果不存在）
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

# 遍历源文件夹中的图片和标签
for filename in os.listdir(source_image_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(source_image_dir, filename)
        label_path = os.path.join(source_label_dir, filename.replace('.png', '.txt'))
        #不用RGB
        image = Image.open(img_path).convert('L')

        # 对图片进行转换操作
        transformed_image = transform(image)
        no_color_image = no_color_transform(image)

        # 读取标签并进行相同的转换操作
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # 将原始图片、转换后的图片和标签保存到目标文件夹
        original_image_save_path = os.path.join(target_image_dir, f'original_{filename}')
        transformed_image_save_path = os.path.join(target_image_dir, f'transformed_{filename}')
        no_color_image_save_path = os.path.join(target_image_dir, f'no_color_{filename}')
        original_label_save_path = os.path.join(target_label_dir, f'original_{filename.replace(".png", ".txt")}')
        transformed_label_save_path = os.path.join(target_label_dir, f'transformed_{filename.replace(".png", ".txt")}')
        no_color_label_save_path = os.path.join(target_label_dir, f'no_color_{filename.replace(".png", ".txt")}')

        image.save(original_image_save_path)
        transformed_image_pil = transforms.ToPILImage()(transformed_image)
        transformed_image_pil.save(transformed_image_save_path)
        no_color_image_pil = transforms.ToPILImage()(no_color_image)
        no_color_image_pil.save(no_color_image_save_path)

        with open(original_label_save_path, 'w') as f:
            f.writelines(labels)
        with open(transformed_label_save_path, 'w') as f:
            f.writelines(labels)
        with open(no_color_label_save_path, 'w') as f:
            f.writelines(labels)

print("All images and labels have been processed and saved.")