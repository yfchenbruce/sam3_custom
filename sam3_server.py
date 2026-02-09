import os
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import fastapi
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import io

# Import SAM3-related modules
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Initialize FastAPI app (enable image preview support in Swagger UI)
app = FastAPI(
    title="SAM3 Segmentation Server",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}  # Hide redundant model expansion
)

# Global variables: model and processor (loaded at server startup)
model = None
processor = None

# Model paths (consistent with infer.py)
BPE_PATH = "/data/chenyifan/sam3/root/.cache/modelscope/hub/models/facebook/sam3/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT_PATH = "/data/chenyifan/sam3/root/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"

# Load model at service startup
@app.on_event("startup")
def load_model():
    global model, processor
    # Load model (disable HF online access, use local files)
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        load_from_HF=False,
        checkpoint_path=CHECKPOINT_PATH,
        compile=False
    )
    processor = Sam3Processor(model)
    print("SAM3 model loaded successfully")

# Reusable visualization function (from infer.py)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor='red',
            facecolor=(0, 0, 0, 0),
            linewidth=2
        )
    )

# Generate visualization image (return BytesIO object)
def generate_vis_image(img: Image.Image, masks, boxes, scores, prompt):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    if len(masks) > 0:
        show_mask(masks[0], plt.gca())
        show_box(boxes[0], plt.gca())
        plt.text(
            boxes[0][0],
            boxes[0][1] - 10,
            f"Score: {scores[0]:.3f}",
            color='white',
            bbox=dict(facecolor='red', alpha=0.8),
            fontsize=12
        )
    plt.axis('off')
    plt.title(f"SAM3 Segmentation: {prompt}", fontsize=14)

    # Save figure to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    return buf

# ========== Core API: return visualization image directly (web preview supported) ==========
@app.post("/segment_vis", summary="Segment image and return visualization (direct preview)")
async def segment_vis_image(
    image: UploadFile = File(..., description="Upload image to be segmented"),
    prompt: str = Form(..., description="Text prompt for segmentation ")
):
    # 1. Read and preprocess image
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    # 2. Model inference
    with torch.no_grad():  # Disable gradient computation for faster inference
        inference_state = processor.set_image(img)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # 3. Process outputs (Tensor to NumPy)
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    if torch.cuda.is_available():
        masks = [m.cpu().numpy() for m in masks]
        boxes = [b.cpu().numpy().tolist() for b in boxes]
        scores = [s.cpu().numpy().item() for s in scores]
    else:
        masks = [m.numpy() for m in masks]
        boxes = [b.numpy().tolist() for b in boxes]
        scores = [s.numpy().item() for s in scores]

    # 4. Generate visualization image and return
    vis_buf = generate_vis_image(img, masks, boxes, scores, prompt)
    return StreamingResponse(
        vis_buf,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=seg_result_{prompt}.png"}
    )

# Start service
if __name__ == "__main__":
    import uvicorn
    print("Starting SAM3 server...")
    uvicorn.run(
        app="sam3_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )