from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re

app = Flask(__name__)
CORS(app)

# === PEMETAAN SIMBOL KE HURUF/ANGKA ===
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

def clean_ocr_text(raw_text):
    if not raw_text:
        return ''
    
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned = ''
        for char in line:
            if char.isalnum() or char.isspace():
                cleaned += char.upper() if char.isalpha() else char
            elif char in symbol_map:
                cleaned += symbol_map[char]
            # simbol lain diabaikan
        # Hapus spasi berlebih di awal/akhir baris
        cleaned = cleaned.strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    
    # Gabungkan baris dengan newline, tapi jangan tambah newline kosong
    return '\n'.join(cleaned_lines)

def levenshtein_distance(s1, s2):
    s1 = s1.replace(' ', '').replace('\n', '')
    s2 = s2.replace(' ', '').replace('\n', '')
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

@app.route('/convert-image', methods=['POST'])
def convert_image():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'text': '', 'similarity': 0})

        image_data = data['image']
        correct = data.get('correct', '')

        ocr_response = requests.post(
            'http://127.0.0.1:3000/ocr',
            json={'image': image_data},
            timeout=15
        )

        if ocr_response.status_code != 200:
            return jsonify({'text': '', 'similarity': 0})

        raw_text = ocr_response.json().get('text', '')
        clean_text = clean_ocr_text(raw_text)

        if not clean_text:
            return jsonify({'text': '(tidak terdeteksi)', 'similarity': 0})

        similarity = 0
        if correct:
            distance = levenshtein_distance(clean_text, correct)
            max_len = max(len(clean_text.replace(' ', '').replace('\n', '')), len(correct.replace(' ', '').replace('\n', '')))
            base_sim = 100 - (distance / max_len) * 100 if max_len > 0 else 100
            len_diff = abs(len(clean_text.replace(' ', '').replace('\n', '')) - len(correct.replace(' ', '').replace('\n', '')))
            penalty = 20 if len_diff > 1 else len_diff * 7
            similarity = round(max(0, base_sim - penalty))

        return jsonify({
            'text': clean_text,
            'similarity': similarity
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'text': '', 'similarity': 0})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
