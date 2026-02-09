import requests
from PIL import Image
import io

SERVER_URL = "http://127.0.0.1:8000/segment_vis"  
IMAGE_PATH = "/data/chenyifan/RF-DETR/_dataset1/test/0-0001_2460.jpg"
PROMPT = "cloth"

try:
    with open(IMAGE_PATH, "rb") as f:
        files = {
            "image": ("test.jpg", f, "image/jpeg")
        }
        data = {
            "prompt": PROMPT
        }

        response = requests.post(
            SERVER_URL,
            files=files,
            data=data,
            timeout=300
        )

    if response.status_code == 200:
        print("Segmentation success")

        img = Image.open(io.BytesIO(response.content))
        img.save("seg_result.png")
        img.show()
    else:
        print(f"Request failed: Status code {response.status_code}")
        print(f"Server response: {response.text}")

except FileNotFoundError:
    print(f"Error: Image file not found at {IMAGE_PATH}")
except requests.exceptions.ConnectionError:
    print(f"Error: Cannot connect to server {SERVER_URL}")
    print("Check if SAM3 server is running and port is correct (8000/8001)")
except requests.exceptions.Timeout:
    print("Error: Request timeout (server may be busy or model inference is slow)")
except Exception as e:
    print(f"Unexpected error: {str(e)}")