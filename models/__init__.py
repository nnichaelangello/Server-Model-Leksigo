# models/__init__.py
from .ocr_model import OCRModel
from .stt_model import STTModel
from .tts_model import TTSModel

# Load sekali saat startup
ocr_model = OCRModel()
stt_model = STTModel()
tts_model = TTSModel()