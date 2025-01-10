import os
import cv2

def txtShow(img, txt, save=True):
    image = cv2.imread(img)
    height, width = image.shape[:2]  # 获取原始图像的高和宽

    # 读取yolo格式标注的txt信息
    with open(txt, 'r') as f:
        labels = f.read().splitlines()

    ob = []  # 存放目标信息
    for i in labels:
        values = i.split(' ')
        if len(values) == 5:
            cl, x_centre, y_centre, w, h = values

            # 需要将数据类型转换成数字型
            cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
            name = f"Class_{cl}"  # 使用类索引作为名称
            xmin = int(x_centre * width - w * width / 2)  # 坐标转换
            ymin = int(y_centre * height - h * height / 2)
            xmax = int(x_centre * width + w * width / 2)
            ymax = int(y_centre * height + h * height / 2)

            tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
            ob.append(tmp)

    # 绘制检测框
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
        cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, color=(0, 0, 255))

    # 创建detect文件夹（如果不存在）
    detect_dir = 'detect'
    os.makedirs(detect_dir, exist_ok=True)

    # 保存图像
    if save:
        save_path = os.path.join(detect_dir, os.path.basename(img))
        cv2.imwrite(save_path, image)

    # 展示图像
    cv2.imshow('test', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    source_image_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\images'
    labels_dir = r'E:\TOOL\datasets\deteset_for_calcification\tarin\labels'

    for filename in os.listdir(source_image_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(source_image_dir, filename)
            label_path = os.path.join(labels_dir, filename.replace('.png', '.txt'))  # 自动获取相应的txt标签文件

            txtShow(img=img_path, txt=label_path, save=True)