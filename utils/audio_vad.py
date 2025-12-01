# utils/audio_vad.py
import webrtcvad
import io
from pydub import AudioSegment

vad = webrtcvad.Vad()
vad.set_mode(3)

def apply_vad(audio_bytes: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    samples = audio.raw_data

    frame_ms = 30
    frame_bytes = 16000 * frame_ms // 1000 * 2
    voiced = []

    for i in range(0, len(samples), frame_bytes):
        frame = samples[i:i+frame_bytes]
        if len(frame) < frame_bytes:
            break
        if vad.is_speech(frame, 16000):
            voiced.append(frame)

    if not voiced:
        return audio_bytes

    voiced_audio = AudioSegment(
        data=b"".join(voiced),
        sample_width=2,
        frame_rate=16000,
        channels=1
    )
    out = io.BytesIO()
    voiced_audio.export(out, format="wav")
    return out.getvalue()