import json
import cv2
import numpy as np
from PIL import Image
import os

def visualize_annotation(image_dir, anno_path, save_dir):
    # 1. 创建保存目录（不存在则自动创建）
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 加载COCO格式标注文件
    with open(anno_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 3. 建立图像id到文件名/图像信息的映射
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 4. 按图像分组标注（避免重复加载同一图像）
    anno_group_by_img = {}
    for anno in coco_data['annotations']:
        img_id = anno['image_id']
        if img_id not in anno_group_by_img:
            anno_group_by_img[img_id] = []
        anno_group_by_img[img_id].append(anno)
    
    # 5. 遍历每张图像，绘制并保存可视化结果
    total_img = len(anno_group_by_img)
    print(f"开始处理，共 {total_img} 张图像需要可视化...")
    
    for idx, (img_id, anno_list) in enumerate(anno_group_by_img.items(), 1):
        # 获取图像信息
        img_info = img_id_to_info[img_id]
        img_file = img_info['file_name']
        img_path = os.path.join(image_dir, img_file)
        
        # 加载图像（处理图像不存在的异常）
        if not os.path.exists(img_path):
            print(f"警告：图像 {img_path} 不存在，跳过该图像")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法加载图像 {img_path}，跳过该图像")
            continue
        
        # 转RGB（保证颜色显示正常，OpenCV默认BGR，PIL默认RGB）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 遍历该图像的所有标注，绘制bbox和mask
        for anno in anno_list:
            # 绘制bbox（COCO格式：[x, y, width, height] → 转(x1,y1,x2,y2)）
            x, y, w, h = anno['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            # 蓝色bbox（RGB格式，线宽2）
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            
            # 绘制半透明mask（处理多边形分割）
            if 'segmentation' in anno and anno['segmentation']:
                for seg in anno['segmentation']:  # 支持多多边形mask
                    seg_points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    # 创建mask画布
                    mask = np.zeros_like(img_rgb)
                    # 填充绿色mask
                    cv2.fillPoly(mask, [seg_points], color=(0, 255, 0))
                    # 叠加半透明效果（原图70%不透明度，mask30%不透明度）
                    img_rgb = cv2.addWeighted(img_rgb, 0.7, mask, 0.3, 0)
        
        # 保存可视化结果
        save_path = os.path.join(save_dir, img_file)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(save_path)
        
        # 打印进度
        print(f"进度：{idx}/{total_img}，已保存 {save_path}")
    
    print(f"所有图像处理完成！结果已保存至 {save_dir}")

# 配置路径（直接使用你提供的路径，无需修改）
IMAGE_DIR = "/data/train_data/hotel/cloth1212/images"
ANNOTATION_FILE = "/data/train_data/hotel/cloth1212/annotaions/instances_default.json"
SAVE_DIR = "/data/chenyifan/sam3/bbox_visualization"

# 调用函数执行可视化
if __name__ == "__main__":
    visualize_annotation(IMAGE_DIR, ANNOTATION_FILE, SAVE_DIR)