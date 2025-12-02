from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from typing import Optional
from google.cloud import vision, speech, texttospeech
import re

# ================= KODE LAMA (SETUP & CLIENTS) =================
vision_client = vision.ImageAnnotatorClient.from_service_account_json("credentials/vision-key.json")
speech_client = speech.SpeechClient.from_service_account_json("credentials/stt-key.json")
tts_client = texttospeech.TextToSpeechClient.from_service_account_json("credentials/tts-key.json")

app = FastAPI(title="AI Game Suite", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=[
    "http://localhost:5173",      # Frontend Local (Vite)
    "http://localhost:3000",      # Backend Local
    "https://leksigo.com",        # Domain Produksi
    "https://www.leksigo.com",    # Domain Produksi (www)
    "https://api.leksigo.com"     # Domain API
], 
allow_credentials=[True], 
allow_methods=["*"], 
allow_headers=["*"])

class OCRRequest(BaseModel):
    image: str
    correct: Optional[str] = None

class STTRequest(BaseModel):
    audio: str
    correct: Optional[str] = None 

class TTSRequest(BaseModel):
    text: str

# ================= KODE BARU (LOGIKA & ALGORITMA) =================

def clean_text(s: str, lowercase: bool = False) -> str:
    """
    Logika Cleaning:
    - Hanya huruf, angka, spasi.
    - lowercase=True -> Ubah ke huruf kecil (Untuk STT)
    - lowercase=False -> Huruf asli (Untuk Gambar)
    - Hapus spasi ganda.
    """
    if not s:
        return ""
    
    cleaned = ''
    for char in s:
        if char.isalnum() or char.isspace():
            if lowercase:
                cleaned += char.lower()
            else:
                cleaned += char  # Huruf asli
            
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def levenshtein(a: str, b: str) -> int:
    """
    Logika Levenshtein:
    - Menggunakan optimasi memori.
    """
    # Hapus spasi untuk perhitungan jarak
    a = a.replace(' ', '')
    b = b.replace(' ', '')

    if len(a) < len(b):
        a, b = b, a

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))

    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def similarity_score(clean: str, target: str, lowercase: bool = False) -> int:
    """
    Logika Similarity:
    - Menerima parameter lowercase agar target (kunci jawaban)
      dibersihkan dengan cara yang SAMA dengan input user.
    """
    if not target:
        return 0
    
    # Target dibersihkan mengikuti mode (STT=lower, Gambar=Asli)
    target_clean = clean_text(target, lowercase=lowercase)
    
    # Siapkan string (hapus spasi)
    s1 = clean.replace(' ', '')
    s2 = target_clean.replace(' ', '')

    dist = levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))

    # Rumus Base Similarity
    base = 100 - (dist / max_len) * 100 if max_len > 0 else 100

    # Rumus Penalty
    len_diff = abs(len(s1) - len(s2))
    penalty = 20 if len_diff > 1 else len_diff * 7

    return round(max(0, base - penalty))

# ================= ENDPOINTS =================

@app.get("/")
def home():
    return {"status": "AI Game Suite API Running"}

@app.post("/image-processing")
def ocr(req: OCRRequest):
    try:
        img = vision.Image(content=base64.b64decode(req.image.split(",")[1]))
        resp = vision_client.text_detection(
            image=img,
            image_context={"language_hints": ["id", "en"]}
        )

        if resp.text_annotations and len(resp.text_annotations) > 0:
            raw = resp.text_annotations[0].description
        else:
            raw = ""

        # GAMBAR: lowercase=False (Gunakan tulisan asli)
        clean = clean_text(raw, lowercase=False)
        
        # Hitung skor (Target juga tidak di-lower)
        sim = similarity_score(clean, req.correct, lowercase=False) if req.correct else None
        
        return {
            "text": clean or "(tidak terdeteksi)",
            "similarity": sim
        }
    except Exception as e:
        raise HTTPException(500, f"OCR Error: {e}")

@app.post("/speech-to-text")
def transcribe(req: STTRequest):
    try:
        audio = speech.RecognitionAudio(content=base64.b64decode(req.audio))
        config = speech.RecognitionConfig(
            encoding="WEBM_OPUS",
            sample_rate_hertz=48000,
            language_code="id-ID"
        )
        resp = speech_client.recognize(config=config, audio=audio)
        raw_text = " ".join([r.alternatives[0].transcript for r in resp.results]) if resp.results else ""
        
        # STT: lowercase=True (Harus lowercase)
        clean = clean_text(raw_text, lowercase=True)

        # Hitung skor (Target juga di-lower agar adil)
        sim = similarity_score(clean, req.correct, lowercase=True) if req.correct else 0

        return {
            "transcription": raw_text,
            "text": clean,
            "similarity": sim
        }
    except Exception as e:
        raise HTTPException(500, f"STT Error: {e}")

@app.post("/text-to-speech")
def tts(req: TTSRequest):
    try:
        if not req.text.strip():
            raise HTTPException(400, "Teks kosong")
        input_ = texttospeech.SynthesisInput(text=req.text)
        voice = texttospeech.VoiceSelectionParams(language_code="id-ID", name="id-ID-Standard-A")
        config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        resp = tts_client.synthesize_speech(input=input_, voice=voice, audio_config=config)
        return {"audioContent": base64.b64encode(resp.audio_content).decode()}
    except Exception as e:
        raise HTTPException(500, f"TTS Error: {e}")
