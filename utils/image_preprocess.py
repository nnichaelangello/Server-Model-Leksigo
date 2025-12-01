# utils/image_preprocess.py
import cv2
import numpy as np
import base64

def preprocess_image(base64_image: str) -> bytes:
    """Grayscale + CLAHE → return PNG bytes"""
    header"""
    header, encoded = base64_image.split(",", 1) if "," in base64_image else ("", base64_image)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    _, buffer = cv2.imencode(".png", enhanced)
    return b"data:image/png;base64," + base64.b64encode(buffer)