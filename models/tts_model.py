# models/tts_model.py
import torch
from TTS.api import TTS
import base64
import tempfile
import os

class TTSModel:
    def __init__(self):
        print("[TTS] Loading model Indonesia (Coqui TTS + HiFi-GAN)...")
        self.model = TTS(
            model_name="tts_models/id/css10/vits",
            progress_bar=False,
            gpu=torch.cuda.is_available()
        )
        print("[TTS] Model Indonesia berhasil dimuat!")

    def predict(self, text: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            self.model.tts_to_file(text=text, file_path=f.name)
            wav_bytes = open(f.name, "rb").read()
            os.unlink(f.name)

        return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode()