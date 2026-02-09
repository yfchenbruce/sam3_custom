import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import binary_dilation

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import plot_results, normalize_bbox

BPE_PATH = "/root/.cache/modelscope/hub/models/facebook/sam3/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/root/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"
CONFIDENCE_THRESHOLD = 0.13
# Mode configuration: text / bbox / text+bbox
INPUT_MODE = "text"  
TEXT_PROMPT = "cloth"
JSON_PATH = "/data/chenyifan/dataset/12/_annotations.coco.json"  # COCO format bbox file

IMAGE_FOLDER = "/data/chenyifan/dataset/12"
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

OUTPUT_ROOT = "/data/chenyifan/sam3/output_crop_12_0.13x15x500"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
CATEGORY_ID = 1

MERGED_COCO_NAME = "cloth_seg_merged_coco.json"

# Post-processing parameters
DILATION_KERNEL_SIZE = 15
CLOSING_KERNEL_SIZE = 5
HOLE_FILL_KERNEL_SIZE = 5
MASK_CENTER_DISTANCE_THRESHOLD = 500
# =====================================================================

class SAM3BatchSegmenter:
    def __init__(self, bpe_path, checkpoint_path, confidence_threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        print("🚀 Initializing SAM3 model...")

        # Load model (initialized only once)
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            load_from_HF=False,
            checkpoint_path=checkpoint_path,
            compile=False
        ).to(self.device)
        self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
        print(
            f"✅ Model initialization complete!\n"
            f"  - Checkpoint: {checkpoint_path}\n"
            f"  - BPE vocabulary: {bpe_path}"
        )

        # Load bbox mapping (filename → bbox list, string matching throughout, no integer conversion)
        self.filename_to_bboxes = self._load_coco_bboxes(JSON_PATH) if INPUT_MODE in ["bbox", "text+bbox"] else {}
        if INPUT_MODE in ["bbox", "text+bbox"]:
            print(f"📋 Loaded bbox data for {len(self.filename_to_bboxes)} images")

        # Initialize COCO structure (image_id as string, avoid integer conversion)
        self.merged_coco = {
            "info": {"description": "SAM3 Cloth Segmentation (Only BBOX Inner)", "version": "1.0", "year": 2025},
            "images": [],
            "categories": [{"id": int(CATEGORY_ID), "name": "cloth", "supercategory": "clothing"}],
            "annotations": []
        }
        self.global_ann_id = 1

        # 新增：用于存储合并后mask数量≠1的图像ID
        self.abnormal_image_ids = {
            "mask_count_0": [],  # 无有效mask
            "mask_count_ge2": []  # mask数量≥2（未完全合并）
        }

    def _load_coco_bboxes(self, json_path):
        """Load COCO JSON: filename → bbox list (no integer ID dependency)"""
        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        
        # 1. image_id (numeric) → filename (string)
        img_id_to_filename = {}
        for img_info in coco_data.get("images", []):
            img_id = img_info["id"]
            if isinstance(img_id, (np.integer, np.floating)):
                img_id = int(img_id) if isinstance(img_id, np.integer) else float(img_id)
            img_id_to_filename[img_id] = img_info["file_name"]
        
        # 2. filename → bbox list (convert to native types)
        filename_to_bboxes = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if isinstance(img_id, (np.integer, np.floating)):
                img_id = int(img_id) if isinstance(img_id, np.integer) else float(img_id)
            
            if img_id in img_id_to_filename:
                filename = img_id_to_filename[img_id]
                if filename not in filename_to_bboxes:
                    filename_to_bboxes[filename] = []
                # Convert bbox to native type list
                bbox = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in ann["bbox"]]
                filename_to_bboxes[filename].append(bbox)
        
        return filename_to_bboxes

    def get_image_unique_id(self, image_path):
        """Extract original image ID (filename without extension, string format)"""
        img_basename = os.path.basename(image_path)
        img_id = os.path.splitext(img_basename)[0]
        return img_id

    def get_all_images_in_folder(self, folder_path):
        """Get all supported image paths"""
        image_paths = []
        for file_name in os.listdir(folder_path):
            if os.path.splitext(file_name)[1].lower() in SUPPORTED_FORMATS:
                image_paths.append(os.path.join(folder_path, file_name))
        image_paths.sort()
        print(f"\n📂 Found {len(image_paths)} images to segment:")
        for idx, path in enumerate(image_paths):
            print(f"   [{idx + 1}] {path}")
        return image_paths

    def fill_mask_holes(self, mask):
        """Fill mask holes"""
        mask_uint8 = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_uint8
        
        max_contour = max(contours, key=cv2.contourArea)
        filled_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(filled_mask, [max_contour], 0, 255, -1)
        
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(filled_mask, [contour], 0, 255, -1)
        
        kernel = np.ones((HOLE_FILL_KERNEL_SIZE, HOLE_FILL_KERNEL_SIZE), np.uint8)
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
        filled_mask = cv2.GaussianBlur(filled_mask, (3, 3), 0)
        return (filled_mask > 127).astype(np.uint8)

    def _is_mask_adjacent(self, mask1, mask2, kernel):
        """Check if masks are adjacent"""
        def get_center(m):
            rows = np.any(m, axis=1)
            cols = np.any(m, axis=0)
            if not np.any(rows) or not np.any(cols):
                return (0.0, 0.0)
            # Convert to native int type
            y1 = int(np.where(rows)[0][0])
            y2 = int(np.where(rows)[0][-1])
            x1 = int(np.where(cols)[0][0])
            x2 = int(np.where(cols)[0][-1])
            return ((x1+x2)/2.0, (y1+y2)/2.0)
        
        cx1, cy1 = get_center(mask1)
        cx2, cy2 = get_center(mask2)
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        
        if distance <= MASK_CENTER_DISTANCE_THRESHOLD:
            return True
        
        dilated1 = binary_dilation(mask1, structure=kernel).astype(np.uint8)
        dilated2 = binary_dilation(mask2, structure=kernel).astype(np.uint8)
        return np.any(np.logical_and(dilated1, dilated2))

    def merge_adjacent_masks(self, masks, scores):
        """Merge adjacent masks"""
        if len(masks) <= 1:
            return [self.fill_mask_holes(m) for m in masks], scores
        
        dilate_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
        closing_kernel = np.ones((CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE), np.uint8)
        
        merge_groups = []
        for mask_idx, mask in enumerate(masks):
            added = False
            for group in merge_groups:
                if any(self._is_mask_adjacent(mask, masks[g_idx], dilate_kernel) for g_idx in group):
                    group.append(mask_idx)
                    added = True
                    break
            if not added:
                merge_groups.append([mask_idx])
        
        merged_masks = []
        merged_scores = []
        for group in merge_groups:
            combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
            for idx in group:
                combined_mask = np.logical_or(combined_mask, masks[idx]).astype(np.uint8)
            
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, closing_kernel)
            combined_mask = self.fill_mask_holes(combined_mask)
            merged_masks.append(combined_mask)
            # Convert to native float type
            merged_scores.append(float(max([scores[idx] for idx in group])))
        
        return merged_masks, merged_scores

    def mask_to_coco_polygon(self, mask):
        """Convert mask to COCO polygon (ensure coordinates are native int)"""
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            # Convert to native int list
            flat = [int(p) for p in contour.flatten().tolist()]
            if len(flat) >= 6:
                polygons.append(flat)
        return polygons

    def get_mask_bbox(self, mask):
        """Extract mask BBOX (ensure coordinates are native int)"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        # Convert to native int type
        y1 = int(np.where(rows)[0][0])
        y2 = int(np.where(rows)[0][-1])
        x1 = int(np.where(cols)[0][0])
        x2 = int(np.where(cols)[0][-1])
        return [x1, y1, x2, y2]

    # -------------------------- 新增：精准裁剪bbox内mask --------------------------
    def crop_mask_to_bbox(self, mask, bboxes, img_w, img_h):
        """强制保留bbox内mask，完全删除外部内容"""
        # 创建全0背景掩码
        bbox_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        # 填充bbox区域为白色（255）
        for bbox in bboxes:
            x1, y1, w, h = map(int, bbox)
            x2 = x1 + w
            y2 = y1 + h
            # 防止超出图像边界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            bbox_mask[y1:y2, x1:x2] = 255
        # 按位与：仅保留bbox内的mask
        return np.logical_and(mask, bbox_mask).astype(np.uint8)

    def add_to_merged_coco(self, image_path, merged_masks, merged_scores, image_id):
        """Add to COCO structure (ensure all values are native types)"""
        image = Image.open(image_path).convert("RGB")
        # Convert to native int type
        img_w = int(image.size[0])
        img_h = int(image.size[1])
        img_name = os.path.basename(image_path)

        # 避免重复添加图像
        if not any(item["id"] == image_id for item in self.merged_coco["images"]):
            self.merged_coco["images"].append({
                "id": image_id,  # String ID
                "file_name": img_name,
                "width": img_w,
                "height": img_h,
                "path": os.path.abspath(image_path)
            })

        for mask, score in zip(merged_masks, merged_scores):
            polygons = self.mask_to_coco_polygon(mask)
            if not polygons:
                continue
            x1, y1, x2, y2 = self.get_mask_bbox(mask)
            # Calculate area and convert to native float
            area = float((x2 - x1) * (y2 - y1))

            self.merged_coco["annotations"].append({
                "id": int(self.global_ann_id),  # Native int
                "image_id": image_id,           # String ID
                "category_id": int(CATEGORY_ID),# Native int
                "bbox": [x1, y1, int(x2 - x1), int(y2 - y1)],  # Native int list
                "area": round(area, 2),         # Native float
                "segmentation": polygons,       # Native int list
                "score": round(float(score), 3),# Native float
                "iscrowd": 0,
                "ignore": 0
            })
            self.global_ann_id += 1

    def save_merged_coco(self):
        """Save COCO file (ensure all values are serializable)"""
        merged_coco_path = os.path.join(OUTPUT_ROOT, MERGED_COCO_NAME)
        with open(merged_coco_path, "w", encoding="utf-8") as f:
            json.dump(self.merged_coco, f, ensure_ascii=False, indent=4)
        print(f"\n📄 Merged COCO JSON saved to: {merged_coco_path}")
        return merged_coco_path

    # 新增：打印合并后mask数量≠1的图像ID统计
    def print_abnormal_image_stats(self):
        print("\n" + "="*50)
        print("📊 合并后mask数量≠1的图像统计")
        print("="*50)
        print(f"1. 无有效mask（mask数量=0）的图像ID：")
        if self.abnormal_image_ids["mask_count_0"]:
            for img_id in self.abnormal_image_ids["mask_count_0"]:
                print(f"   - {img_id}")
        else:
            print("   无此类图像")
        
        print(f"\n2. mask数量≥2（未完全合并）的图像ID：")
        if self.abnormal_image_ids["mask_count_ge2"]:
            for img_id in self.abnormal_image_ids["mask_count_ge2"]:
                print(f"   - {img_id}")
        else:
            print("   无此类图像")
        print("="*50 + "\n")

    def segment_single_image(self, image_path):
        """Process single image (仅保留bbox内分割结果)"""
        img_id = self.get_image_unique_id(image_path)  # String ID
        img_basename = os.path.basename(image_path)
        print(f"\n========== Processing image [{img_id}] ==========")

        # -------------------------- 新增：text+bbox模式下，无BBOX则直接忽略 --------------------------
        if INPUT_MODE == "text+bbox":
            # 提前获取当前图像的BBOX
            current_image_bboxes = self.filename_to_bboxes.get(img_basename, [])
            if not current_image_bboxes:
                print(f"⚠️ {img_basename} 无对应BBOX，在text+bbox模式下直接忽略该图像")
                return False  # 返回False表示处理失败/跳过，不计入成功计数
        # ----------------------------------------------------------------------------------------------

        # Read image
        try:
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size
        except Exception as e:
            print(f"❌ Failed to read image: {image_path} | Error: {e}")
            return False

        # Initialize inference state
        inference_state = self.processor.set_image(image)
        self.processor.reset_all_prompts(inference_state)

        # 保存当前图像的原始bbox
        current_image_bboxes = []

        # 1. Text mode
        if INPUT_MODE in ["text", "text+bbox"]:
            print(f"📝 Using text prompt: {TEXT_PROMPT}")
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt=TEXT_PROMPT
            )

        # 2. BBOX mode
        if INPUT_MODE in ["bbox", "text+bbox"]:
            bboxes = self.filename_to_bboxes.get(img_basename, [])
            current_image_bboxes = bboxes  # 记录当前图像的所有原始bbox
            if not bboxes:
                print(f"⚠️ No bbox found for image {img_basename}, skip bbox prompt")
            else:
                print(f"🔲 Using {len(bboxes)} bbox(es) for image {img_basename}")
                for bbox in bboxes:
                    box_xywh = torch.tensor(bbox).view(-1, 4).to(self.device)
                    box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                    norm_box = normalize_bbox(box_cxcywh, img_w, img_h).flatten().tolist()
                    inference_state = self.processor.add_geometric_prompt(
                        state=inference_state,
                        box=norm_box,
                        label=True
                    )

        # Extract inference results (convert to native types)
        try:
            raw_masks = [m.cpu().numpy().squeeze() for m in inference_state["masks"]]
            # Convert to native float type
            raw_scores = [float(s.cpu().numpy()) for s in inference_state["scores"]]
            valid_idx = [i for i, mask in enumerate(raw_masks) if mask.sum() > 0]
            raw_masks = [raw_masks[i] for i in valid_idx]
            raw_scores = [raw_scores[i] for i in valid_idx]
            print(f"   - Original masks: {len(raw_masks)} | Scores: {[round(s, 3) for s in raw_scores]}")
        except Exception as e:
            print(f"❌ Failed to extract inference results | Error: {e}")
            return False

        # Merge masks
        merged_masks, merged_scores = self.merge_adjacent_masks(raw_masks, raw_scores)
        merged_mask_count = len(merged_masks)
        print(f"   - Merged masks: {merged_mask_count} | Scores: {[round(s, 3) for s in merged_scores]}")

        # -------------------------- 新增：记录mask数量≠1的图像ID --------------------------
        if merged_mask_count == 0:
            self.abnormal_image_ids["mask_count_0"].append(img_id)
        elif merged_mask_count >= 2:
            self.abnormal_image_ids["mask_count_ge2"].append(img_id)
        # ----------------------------------------------------------------------------------------------

        # -------------------------- 替换为精准裁剪：完全删除bbox外内容 --------------------------
        if INPUT_MODE in ["bbox", "text+bbox"] and current_image_bboxes:
            cropped_masks = []
            cropped_scores = []
            for mask, score in zip(merged_masks, merged_scores):
                # 精准裁剪到bbox内
                inner_mask = self.crop_mask_to_bbox(mask, current_image_bboxes, img_w, img_h)
                if inner_mask.sum() > 0:  # 过滤裁剪后为空的mask
                    cropped_masks.append(inner_mask)
                    cropped_scores.append(score)
            # 更新为仅bbox内的结果
            merged_masks = cropped_masks
            merged_scores = cropped_scores
            cropped_mask_count = len(merged_masks)
            print(f"   - BBOX inner masks: {cropped_mask_count} | Scores: {[round(s, 3) for s in merged_scores]}")

            # 裁剪后更新异常图像ID（若裁剪后mask数量变化）
            if cropped_mask_count != merged_mask_count:
                # 移除原合并后mask数量的记录
                if merged_mask_count == 0:
                    if img_id in self.abnormal_image_ids["mask_count_0"]:
                        self.abnormal_image_ids["mask_count_0"].remove(img_id)
                elif merged_mask_count >= 2:
                    if img_id in self.abnormal_image_ids["mask_count_ge2"]:
                        self.abnormal_image_ids["mask_count_ge2"].remove(img_id)
                # 添加裁剪后mask数量的记录
                if cropped_mask_count == 0:
                    self.abnormal_image_ids["mask_count_0"].append(img_id)
                elif cropped_mask_count >= 2:
                    self.abnormal_image_ids["mask_count_ge2"].append(img_id)
        # ----------------------------------------------------------------------------------------------

        # 若无有效mask，直接返回
        if not merged_masks:
            print(f"⚠️ No valid masks left (only bbox inner) for image {img_basename}")
            return True

        # Add to COCO
        self.add_to_merged_coco(image_path, merged_masks, merged_scores, img_id)

        # Save visualization results
        vis_seg_img = os.path.join(OUTPUT_ROOT, f"seg_result_{INPUT_MODE}_{img_id}.png")
        try:
            vis_state = {
                "masks": [torch.tensor(m)[None] for m in merged_masks],
                "scores": [torch.tensor(s) for s in merged_scores],
                "boxes": [torch.tensor(self.get_mask_bbox(m)) for m in merged_masks]
            }
            plot_results(image, vis_state)
            plt.axis("off")
            plt.savefig(vis_seg_img, bbox_inches="tight", pad_inches=0, dpi=100)
            plt.close()
            print(f"✅ BBOX inner segmentation result saved to: {vis_seg_img}")
        except Exception as e:
            print(f"⚠️ Failed to save visualization | Error: {e}")

        return True

    def run_batch_segmentation(self):
        """Batch processing entry point"""
        image_paths = self.get_all_images_in_folder(IMAGE_FOLDER)
        if not image_paths:
            print("\n❌ No supported images found!")
            return

        success_count = 0
        total_count = len(image_paths)
        print(f"\n🚀 Starting batch processing ({total_count} images) in {INPUT_MODE} mode...")
        print(f"🔍 Only keep segmentation results inside BBOX")

        for image_path in image_paths:
            if self.segment_single_image(image_path):
                success_count += 1

        if success_count > 0:
            self.save_merged_coco()

        # 新增：批量处理完成后，打印异常图像统计
        self.print_abnormal_image_stats()

        print("\n========== Batch processing completed ==========")
        print(f"📊 Statistics: {success_count}/{total_count} images processed successfully")
        print(f"📁 Output directory: {OUTPUT_ROOT}")
        if success_count < total_count:
            print(f"⚠️ {total_count - success_count} images failed. Check logs for details.")

if __name__ == "__main__":
    try:
        batch_segmenter = SAM3BatchSegmenter(
            bpe_path=BPE_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        batch_segmenter.run_batch_segmentation()
    except Exception as e:
        print(f"\n❌ Batch processing terminated unexpectedly | Error: {e}")
        import traceback
        traceback.print_exc()