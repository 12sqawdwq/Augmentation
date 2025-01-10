import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import logging
import random

def txtShow(img_path, txt_path, save=True, display=False, detect_dir='detect'):
    """
    在图像上绘制YOLO格式的边界框，并保存或显示图像。

    参数：
    - img_path (str): 图像文件的路径。
    - txt_path (str): 标签文件的路径。
    - save (bool): 是否保存带边界框的图像。
    - display (bool): 是否显示带边界框的图像。
    - detect_dir (str): 保存带边界框图像的目录。
    """
    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        logging.error(f"无法读取图像: {img_path}")
        return
    height, width = image.shape[:2]

    # 读取YOLO格式标签
    if not os.path.exists(txt_path):
        logging.warning(f"标签文件不存在: {txt_path}")
        labels = []
    else:
        with open(txt_path, 'r') as f:
            labels = f.read().splitlines()

    ob = []
    for line in labels:
        values = line.split()
        if len(values) < 5:
            logging.warning(f"{txt_path}中的标签格式无效: {line}")
            continue
        try:
            cl, x_centre, y_centre, w, h = values[:5]
            cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
            name = f"Class_{cl}"
            xmin = int(x_centre * width - w * width / 2)
            ymin = int(y_centre * height - h * height / 2)
            xmax = int(x_centre * width + w * width / 2)
            ymax = int(y_centre * height + h * height / 2)
            ob.append((name, xmin, ymin, xmax, ymax))
        except ValueError as e:
            logging.error(f"解析{txt_path}中的标签时出错: {line} | {e}")
            continue

    # 绘制边界框
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 保存图像
    if save:
        os.makedirs(detect_dir, exist_ok=True)
        save_path = os.path.join(detect_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, image)

    # 显示图像
    if display:
        cv2.imshow('Image with Labels', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        rect = patches.Rectangle((x, y), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
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
        rect = patches.Rectangle((x, y), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x, y - 10, str(cls), bbox=dict(facecolor='yellow', alpha=0.5))

    plt.suptitle(title, fontsize=20)
    plt.show()

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 直接在代码中指定路径
    source_image_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\processed_images_aug_v3\images'
    labels_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\processed_images_aug_v3\labels'
    detect_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\processed_images_aug_v3\detect'  # 保存带有边界框的图像
    visualize = False  # 是否显示图像
    save = True        # 是否保存图像
    visualize_samples = 5  # 选择可视化的样本数量（仅当 visualize=True 时有效）

    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 获取所有图片文件（包括子文件夹）
    image_files = []
    for root, dirs, files in os.walk(source_image_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(root, file))

    if not image_files:
        logging.warning(f"在 {source_image_dir} 中未找到任何图片文件。支持的格式: {supported_formats}")
        return

    # 选择随机样本用于可视化
    if visualize and len(image_files) >= visualize_samples:
        visualize_indices = random.sample(range(len(image_files)), visualize_samples)
    else:
        visualize_indices = []

    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        filename = os.path.basename(img_path)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)

        # 处理图片和标签
        txtShow(img_path, label_path, save=save, display=False, detect_dir=detect_dir)

        # 可视化部分样本
        if visualize and idx in visualize_indices:
            # 读取原始图像和带边界框的图像
            original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            augmented_image = cv2.imread(os.path.join(detect_dir, filename), cv2.IMREAD_GRAYSCALE)
            if augmented_image is None:
                logging.error(f"无法读取增强后的图像: {os.path.join(detect_dir, filename)}")
                continue

            # 读取标签
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.read().splitlines()

            original_boxes = []
            for line in labels:
                values = line.split()
                if len(values) < 5:
                    continue
                try:
                    cl, x_centre, y_centre, w, h = values[:5]
                    cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
                    original_boxes.append([x_centre, y_centre, w, h])
                except ValueError:
                    continue

            # 读取增强后的标签
            augmented_label_path = os.path.join(labels_dir, 'aug_' + label_filename)
            augmented_boxes = []
            augmented_class_labels = []
            if os.path.exists(augmented_label_path):
                with open(augmented_label_path, 'r') as f:
                    augmented_labels = f.read().splitlines()
                for line in augmented_labels:
                    values = line.split()
                    if len(values) < 5:
                        continue
                    try:
                        cl, x_centre, y_centre, w, h = values[:5]
                        cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
                        augmented_class_labels.append(cl)
                        augmented_boxes.append([x_centre, y_centre, w, h])
                    except ValueError:
                        continue

            # 读取增强后的类别标签
            augmented_class_labels = [box[0] for box in augmented_boxes]
            augmented_boxes = [box[1:] for box in augmented_boxes]

            plot_images(
                original_image=original_image,
                augmented_image=augmented_image,
                original_boxes=original_boxes,
                augmented_boxes=augmented_boxes,
                original_class_labels=[f"Class_{box[0]}" for box in original_boxes],
                augmented_class_labels=[f"Class_{cls}" for cls in augmented_class_labels],
                title=f"Augmentation Result for {filename}"
            )

    logging.info("所有图片和标签已处理完毕。")

if __name__ == '__main__':
    main()
