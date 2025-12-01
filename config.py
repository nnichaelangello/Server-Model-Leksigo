# config.py
class Config:
    OCR_ENDPOINT   = None                   
    STT_ENDPOINT   = None                  
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_RATE    = 16000
