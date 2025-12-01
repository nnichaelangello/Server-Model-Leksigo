# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import ocr_model, stt_model, tts_model

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "auto-scoring-api"})

@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"text": ""}), 400
    text = ocr_model.predict(data["image"])
    return jsonify({"text": text})

@app.route("/stt", methods=["POST"])
def stt_endpoint():
    data = request.get_json()
    if not data or "audio" not in data:
        return jsonify({"text": ""}), 400
    text = stt_model.predict(data["audio"])
    return jsonify({"text": text})

@app.route("/tts", methods=["POST"])
def tts_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"audio": ""}), 400
    audio_b64 = tts_model.predict(data["text"])
    return jsonify({"audio": audio_b64})

if __name__ == "__main__":
    print("\nAuto Scoring API berjalan di http://0.0.0.0:5000")
    print("Endpoint:")
    print("  POST /ocr    → Image → Text")
    print("  POST /stt    → Audio → Text")
    print("  POST /tts    → Text  → Audio")
    print("  GET  /health\n")
    app.run(host="0.0.0.0", port=5000, debug=False)