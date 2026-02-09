import json
import os
import numpy as np
import cv2

# ===================== 配置 =====================
INPUT_JSON = "/data/chenyifan/sam3/output/cloth_seg_merged_coco.json"
OUTPUT_JSON = "/data/chenyifan/sam3/output/cloth_seg_merged_coco_cleaned.json"
REMOVED_JSON = "/data/chenyifan/sam3/output/removed_annotations.json"

MIN_AREA = 500          # 最小 mask 面积
MAX_AREA_RATIO = 0.9    # mask 最大占比 image area
MIN_POLY_POINTS = 6     # polygon 最少点数
SCORE_THRESHOLD = 0.5   # 最低置信度（没有 score 的默认通过）
MORPH_KERNEL_SIZE = 5   # 形态学闭操作 kernel
# ===============================================


def mask_from_polygon(segmentation, height, width):
    """polygon -> mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_from_mask(mask):
    """mask -> polygon"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        contour_flat = contour.flatten().tolist()
        if len(contour_flat) >= MIN_POLY_POINTS:
            polygons.append(contour_flat)
    return polygons


def clean_annotations(coco_json):
    cleaned_annotations = []
    removed_annotations = []

    for ann in coco_json["annotations"]:
        remove_reason = None

        # image info
        image_info = next(
            img for img in coco_json["images"]
            if img["id"] == ann["image_id"]
        )
        height, width = image_info["height"], image_info["width"]
        image_area = height * width

        # 1️⃣ score 过滤
        if ann.get("score", 1.0) < SCORE_THRESHOLD:
            remove_reason = "low_score"

        else:
            # 2️⃣ polygon -> mask
            mask = mask_from_polygon(
                ann["segmentation"], height, width
            )

            # 3️⃣ mask 补全
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 4️⃣ 面积过滤
            area = int(mask.sum())
            if area < MIN_AREA:
                remove_reason = f"small_area({area})"
            elif area > MAX_AREA_RATIO * image_area:
                remove_reason = f"large_area({area})"
            else:
                # 5️⃣ mask -> polygon
                polygons = polygon_from_mask(mask)
                if not polygons:
                    remove_reason = "empty_polygon"
                else:
                    # 6️⃣ bbox
                    ys, xs = np.where(mask > 0)
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

                    # 7️⃣ 更新 annotation（保留）
                    ann["segmentation"] = polygons
                    ann["bbox"] = [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1)
                    ]
                    ann["area"] = float(area)

                    cleaned_annotations.append(ann)
                    continue   # ✅ 正常保留

        # ❌ 被删 annotation
        removed_annotations.append({
            "ann_id": ann.get("id"),
            "image_id": ann["image_id"],
            "file_name": image_info["file_name"],
            "reason": remove_reason,
            "segmentation": ann["segmentation"]
        })

    coco_json["annotations"] = cleaned_annotations
    return coco_json, removed_annotations


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        coco_json = json.load(f)

    print(f"Original annotations: {len(coco_json['annotations'])}")

    coco_json, removed_annotations = clean_annotations(coco_json)

    print(f"Cleaned annotations: {len(coco_json['annotations'])}")
    print(f"Removed annotations: {len(removed_annotations)}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(coco_json, f, ensure_ascii=False, indent=4)

    with open(REMOVED_JSON, "w", encoding="utf-8") as f:
        json.dump(removed_annotations, f, ensure_ascii=False, indent=4)

    print(f"Cleaned COCO JSON saved to: {OUTPUT_JSON}")
    print(f"Removed annotations saved to: {REMOVED_JSON}")


if __name__ == "__main__":
    main()
