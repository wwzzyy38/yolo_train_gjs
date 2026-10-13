import os
import cv2
import random
import numpy as np
import shutil
import yaml
from tqdm import tqdm
import multiprocessing
import argparse
import math
from numba import cuda

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

def apply_motion_blur_cuda(image, blur_length, blur_angle):
    """
    CUDA加速的动态模糊
    """
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # 创建模糊核
        kernel = np.zeros((blur_length, blur_length), dtype=np.float32)
        center = blur_length // 2
        angle_rad = np.deg2rad(blur_angle)
        dx = int(np.cos(angle_rad) * center)
        dy = int(np.sin(angle_rad) * center)
        x1, y1 = center - dx, center - dy
        x2, y2 = center + dx, center + dy
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)
        kernel = kernel / np.sum(kernel)
        
        gpu_kernel = cv2.cuda_GpuMat()
        gpu_kernel.upload(kernel)
        
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        
        gpu_result = cv2.cuda.filter2D(gpu_image, -1, gpu_kernel)
        return gpu_result.download()
    else:
        # 回退到CPU版本
        return apply_motion_blur_cpu(image, blur_length, blur_angle)

def apply_motion_blur_cpu(image, blur_length, blur_angle):
    """
    CPU版本的动态模糊
    """
    angle_rad = np.deg2rad(blur_angle)
    kernel = np.zeros((blur_length, blur_length), dtype=np.float32)
    center = blur_length // 2
    dx = int(np.cos(angle_rad) * center)
    dy = int(np.sin(angle_rad) * center)
    x1, y1 = center - dx, center - dy
    x2, y2 = center + dx, center + dy
    cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)
    kernel = kernel / np.sum(kernel)
    
    if len(image.shape) == 3:
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def rotate_and_scale_image_cuda(image, angle, scale):
    """
    CUDA加速的旋转缩放
    """
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        
        gpu_M = cv2.cuda_GpuMat(M)
        gpu_rotated = cv2.cuda.warpAffine(gpu_image, gpu_M, (new_w, new_h))
        
        rotated = gpu_rotated.download()
        return rotated, M, (new_w, new_h)
    else:
        return rotate_and_scale_image_cpu(image, angle, scale)

def rotate_and_scale_image_cpu(image, angle, scale):
    """
    CPU版本的旋转缩放
    """
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
    # 确保bbox有4个元素
    if len(bbox) != 4:
        print(f"警告: bbox格式不正确: {bbox}")
        return [0.5, 0.5, 0.1, 0.1]  # 返回默认值
    
    x_center, y_center, width, height = bbox
    orig_h, orig_w = img_size
    
    # 检查输入值的有效性
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        print(f"警告: 无效的边界框坐标: {bbox}")
        return [0.5, 0.5, 0.1, 0.1]  # 返回一个默认值
    
    abs_x = x_center * orig_w
    abs_y = y_center * orig_h  # 确保在这里定义abs_y
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
    new_y_center = ((new_y1 + new_y2) / new_h) / new_h
    new_width = (new_x2 - new_x1) / new_w
    new_height = (new_y2 - new_y1) / new_h
    new_width = max(0.02, min(1.0, new_width))
    new_height = max(0.02, min(1.0, new_height))
    return [new_x_center, new_y_center, new_width, new_height]

def blend_with_alpha_cuda(background, foreground, x, y):
    """
    CUDA加速的透明混合
    """
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        bg_gpu = cv2.cuda_GpuMat()
        bg_gpu.upload(background)
        
        fg_gpu = cv2.cuda_GpuMat()
        fg_gpu.upload(foreground)
        
        # 使用CUDA混合
        result = cv2.cuda.addWeighted(bg_gpu, 1.0, fg_gpu, 1.0, 0.0)
        return result.download()
    else:
        return blend_with_alpha_cpu(background, foreground, x, y)

def blend_with_alpha_cpu(background, foreground, x, y):
    """
    CPU版本的透明混合
    """
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

def draw_yolo_labels(image, labels, class_names):
    """
    在图像上绘制YOLO格式的标签
    """
    h, w = image.shape[:2]
    for label in labels:
        if len(label) == 5:
            class_id, x_center, y_center, width, height = label
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            
            # 转换为像素坐标
            x = int(x_center * w)
            y = int(y_center * h)
            box_w = int(width * w)
            box_h = int(height * h)
            
            # 计算边界框的左上角和右下角坐标
            x1 = x - box_w // 2
            y1 = y - box_h // 2
            x2 = x + box_w // 2
            y2 = y + box_h // 2
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制类别标签
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def process_one_image(args):
    i, backgrounds, images, labels, class_names, output_dir, draw_labels = args
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
            
            # 随机决定是否对这个对象应用动态模糊（例如30%概率）
            apply_blur = random.random() < 0.3  # 30%概率应用模糊
            original_img = img.copy()  # 保存原始图像用于标签变换
            if apply_blur:
                # 应用动态模糊 - 使用CUDA加速
                blur_length = random.randint(5, 15)  # 模糊长度5-15像素
                blur_angle = random.randint(0, 179)  # 模糊角度0-179度
                
                # 只对RGB通道应用模糊，保留alpha通道
                rgb_channel = img[:, :, :3]
                alpha_channel = img[:, :, 3] if img.shape[2] == 4 else None
                
                blurred_rgb = apply_motion_blur_cuda(rgb_channel, blur_length, blur_angle)
                
                if alpha_channel is not None:
                    # 重新组合RGB和alpha通道
                    img = np.dstack([blurred_rgb, alpha_channel])
                else:
                    img = blurred_rgb

            img_h, img_w = img.shape[:2]
            scale = random.uniform(0.3, 0.8)
            # 使用CUDA加速的旋转缩放
            rotated_img, M, (new_w, new_h) = rotate_and_scale_image_cuda(img, angle, scale)
            
            # 使用原始图像尺寸来变换标签
            try:
                with open(label_path, 'r') as f:
                    bbox_lines = f.readlines()
            except Exception as e:
                print(f"读取标签文件错误: {label_path}, {e}")
                continue
            transformed_bboxes = []
            orig_img_h, orig_img_w = original_img.shape[:2]  # 使用原始图像尺寸
            for line in bbox_lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        bbox = list(map(float, parts[1:]))
                        # 使用原始图像尺寸进行变换
                        transformed_bbox = transform_bbox(bbox, (orig_img_h, orig_img_w), M, (new_w, new_h))
                        transformed_bboxes.append((int(parts[0]), transformed_bbox))
                    except Exception as e:
                        print(f"标签变换错误: {e}, bbox: {bbox}")
                        continue
                else:
                    print(f"标签格式错误: {line.strip()}")
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
                    # 使用CUDA加速的混合
                    bg = blend_with_alpha_cuda(bg, rotated_img, x, y)
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
        
        # 如果启用绘制标签，则在图像上直接绘制
        if draw_labels:
            # 解析标签字符串为数值列表
            label_data = []
            for line in output_labels:
                parts = line.strip().split()
                if len(parts) == 5:
                    label_data.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            
            # 在图像上绘制标签
            bg = draw_yolo_labels(bg, label_data, class_names)
        
        output_img_path = os.path.join(output_dir, "images", split_dir, output_filename + ".jpg")
        output_label_path = os.path.join(output_dir, "labels", split_dir, output_filename + ".txt")
        cv2.imwrite(output_img_path, bg)
        with open(output_label_path, 'w') as f:
            f.writelines(output_labels)
        output_files.append(output_img_path)
    return output_files

def place_images_on_background(backgrounds, images, labels, class_names, output_dir, num_output_images=1000, draw_labels=False):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    all_output_files = []
    args_list = [
        (i, backgrounds, images, labels, class_names, output_dir, draw_labels)
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
    parser = argparse.ArgumentParser(description='YOLO数据增强工具')
    parser.add_argument('--draw_labels', action='store_true', help='在生成的图片上直接绘制YOLO标签')
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_data")
    output_dir = os.path.join(current_dir, "yolo_dataset_1015_2")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"绘制标签: {args.draw_labels}")
    
    # 检查CUDA可用性
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"检测到CUDA设备，可用GPU数量: {cv2.cuda.getCudaEnabledDeviceCount()}")
        print(f"GPU名称: {cv2.cuda.getDeviceName(0)}")
    else:
        print("未检测到CUDA设备，将使用CPU优化版本")
    
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
            backgrounds, image_paths, label_paths, class_names, output_dir, num_output_images=150, draw_labels=args.draw_labels
        )
        create_yaml_file(output_dir, class_names)
        create_class_file(output_dir, class_names)
        print(f"\n数据集生成完成，共生成 {len(all_output_files)} 张图片")
        print(f"数据集已保存到: {output_dir}")
        print(f"YAML配置文件已生成: {os.path.join(output_dir, 'dataset.yaml')}")
        print(f"类别文件已生成: {os.path.join(output_dir, 'labels', 'classes.txt')}")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
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
        print("4. 确保标签文件格式正确 (class_id x_center y_center width height)")

if __name__ == "__main__":
    main()