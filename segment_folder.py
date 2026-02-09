import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import draw_box_on_image, plot_results, normalize_bbox

BPE_PATH = "/root/.cache/modelscope/hub/models/facebook/sam3/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/root/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"
CONFIDENCE_THRESHOLD = 0.4

INPUT_MODE = "text"  # Batch mode: use "text" or "bbox"
TEXT_PROMPT = "stair"  # Text mode: shared prompt for all images
BOX_INPUT_XYWH = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)  # BBOX mode: shared bbox for all images

IMAGE_FOLDER = "/data/train_data/hotel/louti"  # Folder containing images to be segmented
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]  # Supported image formats

OUTPUT_ROOT = "/data/chenyifan/sam3/output_louti"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
CATEGORY_ID = 1  # Category ID for cloth
MERGED_COCO_NAME = "cloth_seg_merged_coco.json"  # Merged COCO JSON filename
# ==================================================================

class SAM3BatchSegmenter:
    def __init__(self, bpe_path, checkpoint_path, confidence_threshold=0.5):
        """Initialize SAM3 model (initialized once and reused for batch processing)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        print("🚀 Initializing SAM3 model...")

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

        # Initialize merged COCO structure (core: store all images and annotations together)
        self.merged_coco = {
            "info": {
                "description": "SAM3 Batch Cloth Segmentation",
                "version": "1.0",
                "year": 2025
            },
            "images": [],
            "categories": [{
                "id": CATEGORY_ID,
                "name": "cloth",
                "supercategory": "clothing"
            }],
            "annotations": []
        }
        self.global_ann_id = 1  # Global annotation ID to avoid duplication

    def get_image_unique_id(self, image_path):
        """Extract unique image ID from file name (used as COCO image.id)"""
        img_basename = os.path.basename(image_path)
        img_id = os.path.splitext(img_basename)[0]
        return img_id

    def get_all_images_in_folder(self, folder_path):
        """Traverse folder and collect all supported image paths"""
        image_paths = []
        for file_name in os.listdir(folder_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in SUPPORTED_FORMATS:
                image_paths.append(os.path.join(folder_path, file_name))

        image_paths.sort()
        print(f"\n📂 Found {len(image_paths)} images to segment:")
        for idx, path in enumerate(image_paths):
            print(f"   [{idx + 1}] {path}")
        return image_paths

    def mask_to_coco_polygon(self, mask):
        """Convert binary mask to COCO polygon format"""
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            contour_flat = contour.flatten().tolist()
            if len(contour_flat) >= 6:
                polygons.append(contour_flat)
        return polygons

    def add_to_merged_coco(self, image_path, masks, boxes, scores, image_id):
        """Append image and annotation info to the merged COCO structure"""
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        img_name = os.path.basename(image_path)

        self.merged_coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": img_w,
            "height": img_h,
            "path": os.path.abspath(image_path)
        })

        for mask, box, score in zip(masks, boxes, scores):
            polygons = self.mask_to_coco_polygon(mask)
            if not polygons:
                continue

            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            ann = {
                "id": self.global_ann_id,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": round(area, 2),
                "segmentation": polygons,
                "score": round(score, 3),
                "iscrowd": 0,
                "ignore": 0
            }
            self.merged_coco["annotations"].append(ann)
            self.global_ann_id += 1

    def save_merged_coco(self):
        """Save merged COCO JSON file"""
        merged_coco_path = os.path.join(OUTPUT_ROOT, MERGED_COCO_NAME)
        with open(merged_coco_path, "w", encoding="utf-8") as f:
            json.dump(self.merged_coco, f, ensure_ascii=False, indent=4)
        print(f"\n📄 Merged COCO JSON saved to: {merged_coco_path}")
        return merged_coco_path

    def segment_single_image(self, image_path):
        """Process a single image (supports text or bbox mode)"""
        img_id = self.get_image_unique_id(image_path)
        print(f"\n========== Processing image [{img_id}] ==========")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Failed to read image: {image_path} | Error: {e}")
            return False

        if INPUT_MODE == "text":
            print(f"📝 Using text prompt: {TEXT_PROMPT}")
            inference_state = self.processor.set_image(image)
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt=TEXT_PROMPT
            )
            vis_seg_img = os.path.join(OUTPUT_ROOT, f"seg_result_text_{img_id}.png")

        elif INPUT_MODE == "bbox":
            print(f"🔲 Using BBOX: {BOX_INPUT_XYWH.tolist()}")
            img_w, img_h = image.size
            inference_state = self.processor.set_image(image)

            box_cxcywh = box_xywh_to_cxcywh(BOX_INPUT_XYWH.to(self.device))
            norm_box = normalize_bbox(box_cxcywh, img_w, img_h).flatten().tolist()

            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.add_geometric_prompt(
                state=inference_state,
                box=norm_box,
                label=True
            )
            vis_seg_img = os.path.join(OUTPUT_ROOT, f"seg_result_bbox_{img_id}.png")

        else:
            print(f"❌ Unsupported input mode: {INPUT_MODE}")
            return False

        try:
            masks = [m.cpu().numpy().squeeze() for m in inference_state["masks"]]
            boxes = [b.cpu().numpy().tolist() for b in inference_state["boxes"]]
            scores = [s.cpu().numpy().item() for s in inference_state["scores"]]
        except Exception as e:
            print(f"❌ Failed to extract inference results | Error: {e}")
            return False

        self.add_to_merged_coco(image_path, masks, boxes, scores, img_id)

        try:
            plot_results(image, inference_state)
            plt.axis("off")
            plt.savefig(vis_seg_img, bbox_inches="tight", pad_inches=0, dpi=100)
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to save segmentation visualization | Error: {e}")

        print("✅ Image processed successfully!")
        print(f"   - Segmentation result image: {vis_seg_img}")
        return True

    def run_batch_segmentation(self):
        """Batch processing entry point"""
        image_paths = self.get_all_images_in_folder(IMAGE_FOLDER)
        if not image_paths:
            print("\n❌ No supported images found!")
            return

        success_count = 0
        total_count = len(image_paths)
        print(f"\n🚀 Starting batch processing ({total_count} images)...")

        for image_path in image_paths:
            if self.segment_single_image(image_path):
                success_count += 1

        if success_count > 0:
            self.save_merged_coco()

        print("\n========== Batch processing completed ==========")
        print(f"📊 Statistics: {success_count}/{total_count} images processed successfully")
        print(f"📁 Output directory: {OUTPUT_ROOT}")
        print(f"📦 Merged COCO JSON: {os.path.join(OUTPUT_ROOT, MERGED_COCO_NAME)}")

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
