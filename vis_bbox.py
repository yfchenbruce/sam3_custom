import cv2
import json
import os
import random

# ===================== 配置参数 =====================
JSON_PATH = "/data/train_data/hotel/cloth1212/annotaions/instances_default.json"  # 标注文件路径
IMAGE_DIR = "/data/train_data/hotel/cloth1212/images"  # 图片目录
OUTPUT_DIR = "/data/chenyifan/sam3/bbox_visualization"  # 可视化结果保存目录
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]  # 支持的图片格式
# ====================================================

def load_coco_annotations(json_path):
    """加载COCO格式标注，构建：image_id→(图片名, bbox列表, 类别名列表)"""
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    
    # 1. 构建类别id→类别名的映射
    cat_id_to_name = {}
    for cat in coco_data.get("categories", []):
        cat_id_to_name[cat["id"]] = cat["name"]
    
    # 2. 构建image_id→图片名/尺寸的映射
    image_id_to_info = {}
    for img in coco_data.get("images", []):
        image_id_to_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        }
    
    # 3. 构建image_id→bbox+类别名的映射
    image_id_to_annos = {}
    for ann in coco_data.get("annotations", []):
        image_id = ann["image_id"]
        bbox = ann["bbox"]  # [x1, y1, w, h]
        cat_name = cat_id_to_name.get(ann["category_id"], "unknown")
        
        if image_id not in image_id_to_annos:
            image_id_to_annos[image_id] = []
        image_id_to_annos[image_id].append({
            "bbox": bbox,
            "category": cat_name
        })
    
    # 合并：image_id→(file_name, annos)
    image_anno_map = {}
    for image_id, info in image_id_to_info.items():
        image_anno_map[info["file_name"]] = image_id_to_annos.get(image_id, [])
    
    return image_anno_map

def get_random_color():
    """生成随机RGB颜色（用于区分不同bbox）"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def visualize_bbox_on_image(image_path, annos, output_path):
    """在单张图片上绘制所有bbox并保存"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 读取图片失败：{image_path}")
        return False
    
    # 遍历所有标注，绘制bbox和类别名
    for idx, anno in enumerate(annos):
        bbox = anno["bbox"]
        cat_name = anno["category"]
        x1, y1, w, h = bbox
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        
        # 生成随机颜色（同一类别可固定颜色，这里简化为随机）
        color = get_random_color()
        
        # 绘制bbox（线宽2）
        cv2.rectangle(img, (int(x1), int(y1)), (x2, y2), color, 2)
        
        # 绘制类别名（背景半透明，避免遮挡）
        text = f"{cat_name}_{idx+1}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_w, text_h = text_size
        # 绘制文本背景
        cv2.rectangle(img, (int(x1), int(y1)-20), (int(x1)+text_w, int(y1)), color, -1)
        # 绘制文本（白色）
        cv2.putText(img, text, (int(x1), int(y1)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 保存图片
    save_success = cv2.imwrite(output_path, img)
    if save_success:
        print(f"✅ 可视化结果已保存：{output_path}")
        return True
    else:
        print(f"❌ 保存图片失败：{output_path}")
        return False

def main():
    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 输出目录：{OUTPUT_DIR}")
    
    # 2. 加载标注数据
    print("📥 加载COCO标注文件...")
    image_anno_map = load_coco_annotations(JSON_PATH)
    print(f"✅ 加载完成，共关联 {len(image_anno_map)} 张图片的标注")
    
    # 3. 遍历图片目录，可视化每张图片的bbox
    success_count = 0
    total_count = 0
    for file_name in os.listdir(IMAGE_DIR):
        # 过滤支持的图片格式
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            continue
        
        total_count += 1
        image_path = os.path.join(IMAGE_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}_with_bbox{file_ext}")
        
        # 获取当前图片的标注
        annos = image_anno_map.get(file_name, [])
        if not annos:
            print(f"⚠️ 图片 {file_name} 无标注，跳过")
            continue
        
        # 可视化bbox
        if visualize_bbox_on_image(image_path, annos, output_path):
            success_count += 1
    
    # 4. 输出统计结果
    print("\n========== 可视化完成 ==========")
    print(f"📊 总计处理图片：{total_count}")
    print(f"✅ 成功可视化：{success_count}")
    print(f"❌ 失败/无标注：{total_count - success_count}")
    print(f"📁 结果目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()