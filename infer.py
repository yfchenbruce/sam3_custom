import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ========== 适配官方环境 + 禁用HF云端 ==========
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

#################################### For Image ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. 加载模型（适配本地文件，避免HF请求）
model = build_sam3_image_model(
    bpe_path="/data/chenyifan/sam3/root/.cache/modelscope/hub/models/facebook/sam3/bpe_simple_vocab_16e6.txt.gz",
    load_from_HF=False,
    checkpoint_path="/data/chenyifan/sam3/root/.cache/modelscope/hub/models/facebook/sam3/sam3.pt",
    compile=False
)
processor = Sam3Processor(model)

# 2. 加载图片（替换为你的路径）
IMAGE_PATH = "/data/chenyifan/RF-DETR/data/_images/clothes_1_capture_0_rgb.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")
inference_state = processor.set_image(image)

# 3. 文本提示分割（替换为你的提示词）
TEXT_PROMPT = "white cloth"
output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

# 4. 解析结果（官方格式：CUDA张量转CPU+NumPy）
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

# ========== 官方风格可视化（SAM3仓库同款实现） ==========
def show_mask(mask, ax, random_color=False):
    """SAM3官方掩码可视化函数"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # 蓝色半透明
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    """SAM3官方边界框可视化函数"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), linewidth=2))

# 转换张量（官方处理逻辑：CPU + NumPy）
if torch.cuda.is_available():
    masks = [m.cpu().numpy() for m in masks]
    boxes = [b.cpu().numpy() for b in boxes]
    scores = [s.cpu().numpy() for s in scores]

# 绘制可视化结果（官方布局）
plt.figure(figsize=(10, 10))
plt.imshow(image)
if len(masks) > 0:
    # 绘制第一个目标（官方示例默认展示top1）
    show_mask(masks[0], plt.gca())
    show_box(boxes[0], plt.gca())
    # 标注置信度（官方风格）
    plt.text(boxes[0][0], boxes[0][1]-10, f"Score: {scores[0]:.3f}", color='white', 
             bbox=dict(facecolor='red', alpha=0.8), fontsize=12)
plt.axis('off')
plt.title(f"SAM3 Segmentation: {TEXT_PROMPT}", fontsize=14)

# 保存结果（官方默认保存为PNG）
save_path = "sam3_segmentation_result.png"
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"官方风格可视化结果已保存至：{save_path}")