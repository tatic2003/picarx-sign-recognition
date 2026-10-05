from flask import Flask, request
import winsound
import tempfile
import os

app = Flask(__name__)

@app.route('/play', methods=['POST'])
def play():
    audio_bytes = request.data
    tmp_path = os.path.join(tempfile.gettempdir(), "tts_recibido.wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
    return "ok"

if __name__ == "__main__":
    print("Escuchando audio en el puerto 5001...")
    app.run(host="0.0.0.0", port=5001)