import os
import cv2
import random
import numpy as np
import shutil
import yaml
from tqdm import tqdm
import multiprocessing

def load_images_and_labels(root_dir):
    backgrounds = []
    image_paths = []
    label_paths = []
    class_names = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")

    required_subdirs = ["background", "label"]
    for subdir in required_subdirs:
        if not os.path.exists(os.path.join(root_dir, subdir)):
            raise FileNotFoundError(f"必需的子目录 '{subdir}' 不存在于 {root_dir}")

    classes_txt_path = os.path.join(root_dir, "label", "classes.txt")
    if os.path.exists(classes_txt_path):
        with open(classes_txt_path, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"未找到类别文件: {classes_txt_path}")

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        if subdir == "background":
            for bg_file in os.listdir(subdir_path):
                if bg_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    bg_img = cv2.imread(os.path.join(subdir_path, bg_file))
                    if bg_img is not None:
                        backgrounds.append(bg_img)
            if not backgrounds:
                print("警告: 没有找到背景图片")
        elif subdir == "label":
            continue
        else:
            for img_file in os.listdir(subdir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir_path, img_file)
                    label_path = os.path.join(root_dir, "label", os.path.splitext(img_file)[0] + ".txt")
                    if os.path.exists(label_path):
                        image_paths.append(img_path)
                        label_paths.append(label_path)
                    else:
                        print(f"警告: 图片 {img_file} 没有对应的标签文件")

    return backgrounds, image_paths, label_paths, class_names

def rotate_and_scale_image(image, angle, scale):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0)
    )
    return rotated, M, (new_w, new_h)

def transform_bbox(bbox, img_size, transform_matrix, new_img_size):
    x_center, y_center, width, height = bbox
    orig_h, orig_w = img_size
    abs_x = x_center * orig_w
    abs_y = y_center * orig_h
    abs_w = width * orig_w
    abs_h = height * orig_h
    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2
    points = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1]
    ])
    transformed_points = np.dot(points, transform_matrix.T)
    new_x1 = np.min(transformed_points[:, 0])
    new_y1 = np.min(transformed_points[:, 1])
    new_x2 = np.max(transformed_points[:, 0])
    new_y2 = np.max(transformed_points[:, 1])
    new_w, new_h = new_img_size
    new_x_center = ((new_x1 + new_x2) / 2) / new_w
    new_y_center = ((new_y1 + new_y2) / 2) / new_h
    new_width = (new_x2 - new_x1) / new_w
    new_height = (new_y2 - new_y1) / new_h
    new_width = max(0.02, min(1.0, new_width))
    new_height = max(0.02, min(1.0, new_height))
    return [new_x_center, new_y_center, new_width, new_height]

def blend_with_alpha(background, foreground, x, y):
    if foreground.shape[2] == 3:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + fg_w)
    y2 = min(bg_h, y + fg_h)
    if x1 >= x2 or y1 >= y2:
        return background
    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)
    fg_roi = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
    fg_alpha = fg_roi[:, :, 3] / 255.0
    fg_alpha = np.expand_dims(fg_alpha, axis=-1)
    bg_roi = background[y1:y2, x1:x2]
    blended = fg_roi[:, :, :3] * fg_alpha + bg_roi * (1 - fg_alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background

def process_one_image(args):
    i, backgrounds, images, labels, class_names, output_dir = args
    output_files = []
    for angle in range(360):
        bg = random.choice(backgrounds).copy()
        bg_h, bg_w = bg.shape[:2]

        # 添加更多随机色块
        num_blocks = random.randint(8, 20)
        for _ in range(num_blocks):
            block_w = random.randint(bg_w // 30, bg_w // 8)
            block_h = random.randint(bg_h // 30, bg_h // 8)
            x1 = random.randint(0, bg_w - block_w)
            y1 = random.randint(0, bg_h - block_h)
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(bg, (x1, y1), (x1 + block_w, y1 + block_h), color, thickness=-1)

        # 50%概率添加随机噪声
        if random.random() < 0.5:
            noise = np.random.randint(0, 50, (bg_h, bg_w, 3), dtype=np.uint8)
            bg = cv2.add(bg, noise)

        num_objects = random.randint(1, min(5, len(images)))
        placed_objects = []
        output_labels = []
        for _ in range(num_objects):
            idx = random.randint(0, len(images)-1)
            img_path = images[idx]
            label_path = labels[idx]
            class_name = os.path.basename(os.path.dirname(img_path))
            class_id = class_names.index(class_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_h, img_w = img.shape[:2]
            scale = random.uniform(0.3, 0.8)
            rotated_img, M, (new_w, new_h) = rotate_and_scale_image(img, angle, scale)
            try:
                with open(label_path, 'r') as f:
                    bbox_lines = f.readlines()
            except Exception:
                continue
            transformed_bboxes = []
            for line in bbox_lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        bbox = list(map(float, parts[1:]))
                        transformed_bbox = transform_bbox(bbox, (img_h, img_w), M, (new_w, new_h))
                        transformed_bboxes.append((int(parts[0]), transformed_bbox))
                    except Exception:
                        continue
            max_attempts = 50
            placed = False
            for _ in range(max_attempts):
                x = random.randint(0, max(1, bg_w - new_w))
                y = random.randint(0, max(1, bg_h - new_h))
                overlap = False
                new_rect = (x, y, x + new_w, y + new_h)
                for rect in placed_objects:
                    if not (new_rect[2] < rect[0] or new_rect[0] > rect[2] or 
                            new_rect[3] < rect[1] or new_rect[1] > rect[3]):
                        overlap = True
                        break
                if not overlap:
                    bg = blend_with_alpha(bg, rotated_img, x, y)
                    placed_objects.append(new_rect)
                    for class_id, bbox in transformed_bboxes:
                        new_x_center = (bbox[0] * new_w + x) / bg_w
                        new_y_center = (bbox[1] * new_h + y) / bg_h
                        new_width = bbox[2] * new_w / bg_w
                        new_height = bbox[3] * new_h / bg_h
                        new_x_center = max(0.0, min(1.0, new_x_center))
                        new_y_center = max(0.0, min(1.0, new_y_center))
                        new_width = max(0.02, min(1.0, new_width))
                        new_height = max(0.02, min(1.0, new_height))
                        output_labels.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
                    placed = True
                    break
            if not placed:
                continue
        output_filename = f"aug_{i:04d}_{angle:03d}"
        is_train = random.random() < 0.8
        split_dir = "train" if is_train else "val"
        output_img_path = os.path.join(output_dir, "images", split_dir, output_filename + ".jpg")
        output_label_path = os.path.join(output_dir, "labels", split_dir, output_filename + ".txt")
        cv2.imwrite(output_img_path, bg)
        with open(output_label_path, 'w') as f:
            f.writelines(output_labels)
        output_files.append(output_img_path)
    return output_files  # 修复：必须返回列表

def place_images_on_background(backgrounds, images, labels, class_names, output_dir, num_output_images=1000):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    all_output_files = []
    args_list = [
        (i, backgrounds, images, labels, class_names, output_dir)
        for i in range(num_output_images)
    ]
    cpu_count = min(multiprocessing.cpu_count(), 24)
    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, args_list), total=num_output_images, desc="多进程增强数据"):
            if result:  # 防止NoneType
                all_output_files.extend(result)
    return all_output_files

def create_yaml_file(output_dir, class_names):
    yaml_content = {
        'train': os.path.join('.', 'images', 'train'),
        'val': os.path.join('.', 'images', 'val'),
        'nc': len(class_names),
        'names': class_names
    }
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

def create_class_file(output_dir, class_names):
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    class_file_path = os.path.join(label_dir, "classes.txt")
    with open(class_file_path, 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}\n")
    print(f"已创建类别文件: {class_file_path}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_data")
    output_dir = os.path.join(current_dir, "yolo_dataset_927")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    try:
        backgrounds, image_paths, label_paths, class_names = load_images_and_labels(input_dir)
        if not backgrounds:
            print("错误：没有找到背景图片")
            return
        if not image_paths:
            print("错误：没有找到类别图片")
            return
        if not class_names:
            print("错误：没有找到类别文件夹")
            return
        print(f"找到 {len(backgrounds)} 张背景图片")
        print(f"找到 {len(image_paths)} 张类别图片")
        print(f"找到 {len(class_names)} 个类别: {', '.join(class_names)}")
        print("开始生成增强数据集...")
        all_output_files = place_images_on_background(
            backgrounds, image_paths, label_paths, class_names, output_dir, num_output_images=150
        )
        create_yaml_file(output_dir, class_names)
        create_class_file(output_dir, class_names)
        print(f"\n数据集生成完成，共生成 {len(all_output_files)} 张图片")
        print(f"数据集已保存到: {output_dir}")
        print(f"YAML配置文件已生成: {os.path.join(output_dir, 'dataset.yaml')}")
        print(f"类别文件已生成: {os.path.join(output_dir, 'labels', 'classes.txt')}")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print("请检查:")
        print(f"1. 输入目录 {input_dir} 是否存在")
        print("2. 输入目录结构是否正确:")
        print("   input_data/")
        print("   ├── background/       # 存放背景图片")
        print("   ├── label/            # 存放YOLO格式标签(.txt)")
        print("   ├── class1/           # 类别1图片")
        print("   ├── class2/           # 类别2图片")
        print("   └── ...               # 其他类别")
        print("3. 确保所有图片和标签文件都是有效的")

if __name__ == "__main__":
    main()