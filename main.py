import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import random

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_directories(save_path):
    """
    创建目标文件夹（如果不存在）
    """
    images_save_path = os.path.join(save_path, "images")
    labels_save_path = os.path.join(save_path, "labels")
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)
    return images_save_path, labels_save_path


def get_label_file(label_path, image_name):
    """
    根据图片信息，查找对应的label
    """
    base_name = os.path.splitext(image_name)[0]
    fname = os.path.join(label_path, base_name + ".txt")
    data2 = []
    if not os.path.exists(fname):
        logging.warning(f"Label file {fname} does not exist.")
        return data2
    if os.path.getsize(fname) == 0:
        logging.warning(f"Label file {fname} is empty.")
        return data2
    else:
        with open(fname, 'r', encoding='utf-8') as infile:
            # 读取并转换标签
            for line_num, line in enumerate(infile, 1):
                data_line = line.strip().split()
                if len(data_line) < 5:
                    logging.warning(f"Line {line_num} in {fname} has fewer than 5 values. Skipping.")
                    continue
                elif len(data_line) > 5:
                    logging.warning(f"Line {line_num} in {fname} has more than 5 values. Extra values will be ignored.")
                # 只取前5个值
                cls, x_c, y_c, w, h = data_line[:5]
                try:
                    data2.append({
                        'class': int(cls),
                        'x_center': float(x_c),
                        'y_center': float(y_c),
                        'width': float(w),
                        'height': float(h)
                    })
                except ValueError as ve:
                    logging.error(f"Line {line_num} in {fname} has invalid values: {line.strip()}")
                    logging.error(f"Exception: {ve}")
    return data2


def save_Yolo(img, boxes, images_save_path, labels_save_path, prefix, image_name):
    """
    将增强后的图像和标签保存到指定路径
    """
    try:
        # 将Tensor转换为NumPy数组
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        # 灰度图像只有一个通道，确保正确保存
        if len(img_np.shape) == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)
            cv2.imwrite(os.path.join(images_save_path, prefix + image_name), img_np)
        else:
            cv2.imwrite(os.path.join(images_save_path, prefix + image_name), img_np)

        # 保存标签
        label_file = os.path.join(labels_save_path, prefix + os.path.splitext(image_name)[0] + ".txt")
        with open(label_file, 'w', encoding="utf-8") as f:
            for box in boxes:
                cls = box['class']
                x_center = box['x_center']
                y_center = box['y_center']
                width = box['width']
                height = box['height']
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
    except Exception as e:
        logging.error(f"ERROR: Failed to save image and labels for {image_name}. Exception: {e}")


def plot_images(original_image, augmented_image, original_boxes, augmented_boxes, original_class_labels, augmented_class_labels, title='Augmentation Result'):
    """
    显示原始图像和增强后的图像，并绘制边界框
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 原始图像
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    for cls, box in zip(original_class_labels, original_boxes):
        x_center, y_center, w, h = box
        x = (x_center - w / 2) * original_image.shape[1]
        y = (y_center - h / 2) * original_image.shape[0]
        box_w = w * original_image.shape[1]
        box_h = h * original_image.shape[0]
        rect = plt.Rectangle((x, y), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x, y - 10, str(cls), bbox=dict(facecolor='yellow', alpha=0.5))

    # 增强后的图像
    augmented_image_np = augmented_image.permute(1, 2, 0).cpu().numpy()
    augmented_image_np = (augmented_image_np * 255).astype(np.uint8)
    if len(augmented_image_np.shape) == 3 and augmented_image_np.shape[2] == 1:
        augmented_image_np = augmented_image_np.squeeze(axis=2)
    axes[1].imshow(augmented_image_np, cmap='gray')
    axes[1].set_title('Augmented Image')
    axes[1].axis('off')
    for cls, box in zip(augmented_class_labels, augmented_boxes):
        x_center, y_center, w, h = box
        x = (x_center - w / 2) * augmented_image_np.shape[1]
        y = (y_center - h / 2) * augmented_image_np.shape[0]
        box_w = w * augmented_image_np.shape[1]
        box_h = h * augmented_image_np.shape[0]
        rect = plt.Rectangle((x, y), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x, y - 10, str(cls), bbox=dict(facecolor='yellow', alpha=0.5))

    plt.suptitle(title, fontsize=20)
    plt.show()


def runAugmentation(image_path, label_path, save_path, visualize=False, visualize_samples=5):
    """
    执行数据增强
    """
    images_save_path, labels_save_path = create_directories(save_path)

    # 获取所有图片文件（仅灰度图像）
    image_list = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    if not image_list:
        logging.warning(f"No images found in {image_path}.")
        return

    # 选择随机样本用于可视化
    visualize_indices = random.sample(range(len(image_list)), min(visualize_samples, len(image_list))) if visualize else []

    for idx, image_name in enumerate(tqdm(image_list, desc="Augmenting images")):
        logging.info(f"Processing: {image_name}")
        img_path = os.path.join(image_path, image_name)
        # 以灰度模式读取图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logging.error(f"ERROR: Unable to read image {image_name}. Skipping.")
            continue
        # 将灰度图像转换为RGB格式（单通道）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 读取标签
        labels = get_label_file(label_path, image_name)
        if labels:
            bboxes = [[label['x_center'], label['y_center'], label['width'], label['height']] for label in labels]
            class_labels = [label['class'] for label in labels]
        else:
            bboxes = []
            class_labels = []

        # 打印transform的类型以调试
        # 这一步可以帮助确认transform是否被意外覆盖为列表
        print(f"Type of transform: {type(transform)}")

        # 如果没有标签，Albumentations 仍然可以处理
        try:
            augmented = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            logging.error(f"ERROR: Augmentation failed for image {image_name}. Exception: {e}")
            continue

        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_labels = augmented['class_labels']

        # 将增强后的图像和标签保存
        prefix = "aug_"
        augmented_boxes = [{'class': cls, 'x_center': x, 'y_center': y, 'width': w, 'height': h}
                           for cls, (x, y, w, h) in zip(augmented_class_labels, augmented_bboxes)]
        save_Yolo(augmented_image, augmented_boxes, images_save_path, labels_save_path, prefix, image_name)

        # 可视化部分样本
        if visualize and idx in visualize_indices:
            plot_images(
                original_image=image,
                augmented_image=augmented_image,
                original_boxes=bboxes,
                augmented_boxes=augmented_bboxes,
                original_class_labels=class_labels,
                augmented_class_labels=augmented_class_labels,
                title=f"Augmentation Result for {image_name}"
            )

    logging.info("All images and labels have been processed and saved.")


# 定义数据增强的Albumentations转换（仅灰度图像）
transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.5),  # 饱和度和色调为0，因为是灰度图
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


if __name__ == '__main__':
    # 图像和标签文件夹路径
    image_path = r"E:\TOOL\datasets\deteset_for_calcification\tarin\images"
    label_path = r"E:\TOOL\datasets\deteset_for_calcification\tarin\labels"
    save_path = r"E:\TOOL\datasets\deteset_for_calcification\tarin\processed_images_aug_v3"  # 结果保存位置路径

    # 是否可视化增强效果
    visualize = True
    visualize_samples = 5  # 选择可视化的样本数量

    # 运行
    runAugmentation(image_path, label_path, save_path, visualize=visualize, visualize_samples=visualize_samples)