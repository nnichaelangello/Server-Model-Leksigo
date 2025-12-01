# models/ocr_model.py
import base64
import requests
from utils.image_preprocess import preprocess_image

EXTERNAL_OCR = None  # "http://localhost:3000/ocr"

class OCRModel:
    def __init__(self):
        print("[OCR] Model siap (menggunakan preprocessing internal + external server)")

    def predict(self, base64_image: str) -> str:
        processed_b64 = preprocess_image(base64_image).decode()

        if EXTERNAL_OCR:
            resp = requests.post(EXTERNAL_OCR, json={"image": processed_b64}, timeout=15)
            return resp.json().get("text", "(gagal)").strip()