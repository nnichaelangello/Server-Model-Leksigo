# models/stt_model.py
import base64
import io
import requests
from utils.audio_vad import apply_vad

EXTERNAL_STT = None  # "http://localhost:9000/v1/audio/transcriptions"

class STTModel:
    def __init__(self):
        print("[STT] Whisper + VAD siap")

    def predict(self, base64_audio: str) -> str:
        audio_bytes = base64.b64decode(base64_audio.split(",", 1)[1] if "," in base64_audio else base64_audio)
        clean_wav = apply_vad(audio_bytes)

        if EXTERNAL_STT:
            files = {"file": ("audio.wav", clean_wav, "audio/wav")}
            data = {"model": "whisper-1"}
            resp = requests.post(EXTERNAL_STT, files=files, data=data, timeout=30)
            return resp.json().get("text", "(kosong)").strip()