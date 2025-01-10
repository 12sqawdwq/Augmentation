import os
from PIL import Image
from torchvision import transforms
import torch
from torchvision.transforms import autoaugment

# 定义标签坐标转换函数
def transform_label(x_centre, y_centre, w, h, original_image, transformed_image):
    # 获取原始图片和变换后图片的尺寸
    original_width, original_height = original_image.size
    transformed_width, transformed_height = transformed_image.shape[2], transformed_image.shape[1]

    # 计算比例
    width_ratio = transformed_width / original_width
    height_ratio = transformed_height / original_height

    # 计算新的标签坐标
    new_x_centre = x_centre * width_ratio
    new_y_centre = y_centre * height_ratio
    new_w = w * width_ratio
    new_h = h * height_ratio

    return new_x_centre, new_y_centre, new_w, new_h

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
        # 标签坐标变化随图片变化
        with open(label_path, 'r') as f:
            labels = f.read().splitlines()

        # 读取标签并进行相同的转换操作
        # 标签坐标变化随图片变化未能成功实现需要进行修改2025/1/10
        new_labels = []
        for label in labels:
            cl, x_centre, y_centre, w, h = label.split(' ')
            cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)

            # 计算新的标签坐标
            # 这里假设你有一个函数 transform_label 来根据图片变换调整标签坐标
            new_x_centre, new_y_centre, new_w, new_h = transform_label(x_centre, y_centre, w, h, image,transformed_image)

            new_labels.append(f"{cl} {new_x_centre} {new_y_centre} {new_w} {new_h}\n")
        # 保存标签
        for i in range(len(labels)):
            cl, x_centre, y_centre, w, h = labels[i].split(' ')
            cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
            labels[i] = f"{cl} {x_centre} {y_centre} {w} {h}\n"


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