from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import requests
import io
import os
import wave
import tempfile
from pydub import AudioSegment
import webrtcvad
from scipy.io import wavfile
import torch
from TTS.api import TTS

app = Flask(__name__)
CORS(app)

# ========================== KONFIGURASI ============================

OCR_ENDPOINT = "http://127.0.0.1:3000/ocr"

STT_ENDPOINT = "http://127.0.0.1:9000/v1/audio/transcriptions"

print("[INFO] Loading TTS model Indonesia (G2P Indonesia + HiFi-GAN)...")
tts = TTS(model_name="tts_models/id/css10/vits", progress_bar=True, gpu=torch.cuda.is_available())

# VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  

# ======================= SYMBOL MAPPING OCR ========================
symbol_map = {
    'O': 'O', '0': 'O', 'o': 'O', 'Q': 'O',
    'I': 'I', '1': 'I', 'l': 'I', '|': 'I', '!': 'I', 'i': 'I',
    'A': 'A', '@': 'A', 'a': 'A', '4': 'A',
    'B': 'B', '8': 'B', 'b': 'B',
    'S': 'S', '5': 'S', '$': 'S',
    'G': 'G', '6': 'G', '9': 'G',
    'Z': 'Z', '2': 'Z', '7': 'Z',
    'T': 'T', '7': 'T',
    'E': 'E', '3': 'E',
    'C': 'C', '(': 'C', '{': 'C', 'c': 'C',
    'U': 'U', 'V': 'U', 'u': 'U',
    'M': 'M', 'W': 'M', 'm': 'M',
    'N': 'N', 'H': 'N',
}

# ======================= FUNGSI UTILITAS ==========================
def clean_ocr_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned = ""
        for char in line:
            if char.isalnum() or char.isspace():
                cleaned += char.upper() if char.isalpha() else char
            elif char in symbol_map:
                cleaned += symbol_map[char]
        cleaned = cleaned.strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)

def levenshtein_distance(s1: str, s2: str) -> int:
    s1 = s1.replace(" ", "").replace("\n", "")
    s2 = s2.replace(" ", "").replace("\n", "")
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_similarity(result: str, correct: str) -> int:
    if not correct or not result:
        return 0
    if result in ["(tidak terdeteksi)", "(diam)", "(error)", ""]:
        return 0
    distance = levenshtein_distance(result, correct)
    max_len = max(len(result.replace(" ", "").replace("\n", "")), len(correct.replace(" ", "")))
    if max_len == 0:
        return 100
    base = 100 - (distance / max_len * 100)
    len_diff = abs(len(result.replace(" ", "")) - len(correct.replace(" ", "")))
    penalty = 25 if len_diff > 2 else len_diff * 8
    return round(max(0, base - penalty))

# ======================= IMAGE PREPROCESSING ======================
def preprocess_image_for_ocr(base64_image: str) -> str:
    """Grayscale + CLAHE → return base64 lagi"""
    img_data = base64_image.split(",")[1] if "," in base64_image else base64_image
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, buffer = cv2.imencode(".png", enhanced)
    enhanced_b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{enhanced_b64}"

# ========================== VAD AUDIO ==============================
def apply_vad_and_get_wav(audio_bytes: bytes) -> bytes:
    """Return WAV 16kHz 16bit mono hanya bagian yang terdeteksi suara"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  
    samples = audio.raw_data

    frame_duration_ms = 30
    frame_bytes = int(16000 * frame_duration_ms / 1000 * 2) 

    voiced_frames = []
    for i in range(0, len(samples), frame_bytes):
        frame = samples[i:i + frame_bytes]
        if len(frame) < frame_bytes:
            break
        if vad.is_speech(frame, 16000):
            voiced_frames.append(frame)

    if not voiced_frames:  
        return audio_bytes

    voiced_raw = b"".join(voiced_frames)
    voiced_audio = AudioSegment(
        data=voiced_raw,
        sample_width=2,
        frame_rate=16000,
        channels=1
    )

    output = io.BytesIO()
    voiced_audio.export(output, format="wav")
    return output.getvalue()

# ======================= ENDPOINT IMAGE → TEXT =====================
@app.route("/convert-image", methods=["POST"])
def convert_image():
    try:
        payload = request.get_json()
        if not payload or "image" not in payload:
            return jsonify({"text": "", "similarity": 0}), 400

        image_b64 = payload["image"]
        correct = payload.get("correct", "").upper()

        processed_b64 = preprocess_image_for_ocr(image_b64)

        ocr_resp = requests.post(OCR_ENDPOINT, json={"image": processed_b64}, timeout=20)
        if ocr_resp.status_code != 200:
            return jsonify({"text": "(OCR server error)", "similarity": 0})

        raw_text = ocr_resp.json().get("text", "")
        clean_text = clean_ocr_text(raw_text).upper()
        if not clean_text.strip():
            clean_text = "(tidak terdeteksi)"

        similarity = calculate_similarity(clean_text, correct)

        return jsonify({
            "text": clean_text,
            "similarity": similarity
        })

    except Exception as e:
        print("[ERROR /convert-image]", e)
        return jsonify({"text": "(error)", "similarity": 0}), 500

# ======================= ENDPOINT AUDIO → TEXT =====================
@app.route("/convert-audio", methods=["POST"])
def convert_audio():
    try:
        payload = request.get_json()
        if not payload or "audio" not in payload:
            return jsonify({"text": "", "similarity": 0}), 400

        audio_b64 = payload["audio"]
        correct = payload.get("correct", "").upper()

        audio_bytes = base64.b64decode(audio_b64.split(",")[1] if "," in audio_b64 else audio_b64)

        clean_wav_bytes = apply_vad_and_get_wav(audio_bytes)

        files = {"file": ("speech.wav", clean_wav_bytes, "audio/wav")}
        data = {"model": "whisper-1"}
        stt_resp = requests.post(STT_ENDPOINT, files=files, data=data, timeout=40)

        if stt_resp.status_code != 200:
            return jsonify({"text": "(STT server error)", "similarity": 0})

        result = stt_resp.json()
        text = result.get("text", "").strip().upper()

        similarity = calculate_similarity(text, correct)

        return jsonify({
            "text": text or "(diam)",
            "similarity": similarity
        })

    except Exception as e:
        print("[ERROR /convert-audio]", e)
        return jsonify({"text": "(error)", "similarity": 0}), 500

# ======================= ENDPOINT TEXT → SPEECH =====================
@app.route("/text-to-speech", methods=["POST"])
def text_to_speech():
    try:
        payload = request.get_json()
        if not payload or "text" not in payload:
            return jsonify({"audio": "", "similarity": 0}), 400

        text = payload["text"].strip()
        correct = payload.get("correct", "").upper() 

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tts.tts_to_file(text=text, file_path=tmp.name)

            with open(tmp.name, "rb") as f:
                wav_bytes = f.read()
            os.unlink(tmp.name)

        audio_base64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode()

        similarity = calculate_similarity(text.upper(), correct) if correct else 0

        return jsonify({
            "audio": audio_base64,
            "similarity": similarity
        })

    except Exception as e:
        print("[ERROR /text-to-speech]", e)
        return jsonify({"audio": "", "similarity": 0}), 500

# ========================== RUN SERVER =============================
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║               SERVER SIAP DIGUNAKAN                      ║
    ║  POST  http://localhost:5000/convert-image                ║
    ║  POST  http://localhost:5000/convert-audio                ║
    ║  POST  http://localhost:5000/text-to-speech               ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
