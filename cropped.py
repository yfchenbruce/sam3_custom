import json
import cv2
import numpy as np
from PIL import Image
import os

def crop_image_and_update_anno(image_dir, anno_path, crop_img_dir, new_anno_path):
    # 1. 创建裁剪图片保存目录
    os.makedirs(crop_img_dir, exist_ok=True)
    
    # 2. 加载原始COCO标注
    with open(anno_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 3. 初始化新标注的结构
    new_coco_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # 4. 建立图像id到图像信息的映射
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 5. 遍历所有标注，逐目标裁剪并更新标注
    new_img_id = 0  # 新图像id（从0开始递增）
    new_anno_id = 0  # 新标注id（从0开始递增）
    total_anno = len(coco_data['annotations'])
    print(f"开始处理，共 {total_anno} 个目标需要裁剪...")
    
    for anno_idx, anno in enumerate(coco_data['annotations'], 1):
        # 获取原始图像信息
        orig_img_id = anno['image_id']
        orig_img_info = img_id_to_info[orig_img_id]
        orig_img_file = orig_img_info['file_name']
        orig_img_path = os.path.join(image_dir, orig_img_file)
        
        # 检查原始图像是否存在
        if not os.path.exists(orig_img_path):
            print(f"警告：原始图像 {orig_img_path} 不存在，跳过该标注")
            continue
        
        # 加载原始图像
        orig_img = cv2.imread(orig_img_path)
        if orig_img is None:
            print(f"警告：无法加载原始图像 {orig_img_path}，跳过该标注")
            continue
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # --------------------------
        # 步骤1：获取原始bbox并裁剪图像
        # --------------------------
        # 原始bbox：COCO格式 [x, y, width, height]
        orig_x, orig_y, orig_w, orig_h = anno['bbox']
        orig_x1, orig_y1 = int(orig_x), int(orig_y)
        orig_x2, orig_y2 = int(orig_x + orig_w), int(orig_y + orig_h)
        
        # 裁剪目标区域（基于原始图像坐标）
        cropped_img_rgb = orig_img_rgb[orig_y1:orig_y2, orig_x1:orig_x2]
        cropped_img_h, cropped_img_w = cropped_img_rgb.shape[:2]
        
        # 跳过裁剪后为空的图像
        if cropped_img_h == 0 or cropped_img_w == 0:
            print(f"警告：标注 {anno_idx} 裁剪后图像为空，跳过")
            continue
        
        # --------------------------
        # 步骤2：更新bbox（适配裁剪后的新图像）
        # --------------------------
        # 裁剪后的bbox：目标在新图像中左上角为(0,0)，宽高不变
        new_bbox = [0.0, 0.0, float(cropped_img_w), float(cropped_img_h)]
        
        # --------------------------
        # 步骤3：更新mask（适配裁剪后的新图像）
        # --------------------------
        new_segmentation = []
        if 'segmentation' in anno and anno['segmentation']:
            for seg in anno['segmentation']:
                # 将原始mask坐标转为(x,y)对，并减去裁剪框的偏移量
                orig_seg_points = np.array(seg).reshape(-1, 2).astype(np.float32)
                # 偏移修正：x减去orig_x1，y减去orig_y1
                new_seg_points = orig_seg_points - np.array([orig_x1, orig_y1])
                # 转为列表并加入新segmentation
                new_seg = new_seg_points.flatten().tolist()
                new_segmentation.append(new_seg)
        
        # --------------------------
        # 步骤4：保存裁剪后的图像
        # --------------------------
        cropped_img_name = f"cropped_{new_img_id}_{orig_img_file}"
        cropped_img_save_path = os.path.join(crop_img_dir, cropped_img_name)
        img_pil = Image.fromarray(cropped_img_rgb)
        img_pil.save(cropped_img_save_path)
        
        # --------------------------
        # 步骤5：添加新图像信息到新标注
        # --------------------------
        new_img_info = {
            "id": new_img_id,
            "file_name": cropped_img_name,
            "width": cropped_img_w,
            "height": cropped_img_h,
            "date_captured": orig_img_info.get("date_captured", ""),
            "license": orig_img_info.get("license", 0)
        }
        new_coco_data['images'].append(new_img_info)
        
        # --------------------------
        # 步骤6：添加新标注信息到新标注
        # --------------------------
        new_anno_info = {
            "id": new_anno_id,
            "image_id": new_img_id,
            "category_id": anno['category_id'],
            "bbox": new_bbox,
            "area": float(cropped_img_w * cropped_img_h),  # 新面积=裁剪后宽×高
            "segmentation": new_segmentation,
            "iscrowd": anno.get("iscrowd", 0)
        }
        new_coco_data['annotations'].append(new_anno_info)
        
        # 打印进度
        print(f"进度：{anno_idx}/{total_anno}，已裁剪并保存 {cropped_img_save_path}")
        
        # 更新id计数器
        new_img_id += 1
        new_anno_id += 1
    
    # --------------------------
    # 步骤7：保存新的COCO标注文件
    # --------------------------
    with open(new_anno_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有处理完成！")
    print(f"1. 裁剪后的图片保存至：{crop_img_dir}")
    print(f"2. 更新后的标注文件保存至：{new_anno_path}")

# 配置路径（使用你提供的基础路径，补充裁剪后路径）
IMAGE_DIR = "/data/chenyifan/RF-DETR/cloth/valid"
ANNOTATION_FILE = "/data/chenyifan/RF-DETR/cloth/valid/_annotations.coco.json"
CROP_IMG_DIR = "/data/chenyifan/RF-DETR/_cloth_cropped/valid"  # 裁剪图片保存目录
NEW_ANNOTATION_FILE = "/data/chenyifan/RF-DETR/_cloth_cropped/valid/_annotations_cropped.coco.json"  # 新标注文件路径

# 执行裁剪和标注更新
if __name__ == "__main__":
    crop_image_and_update_anno(IMAGE_DIR, ANNOTATION_FILE, CROP_IMG_DIR, NEW_ANNOTATION_FILE)