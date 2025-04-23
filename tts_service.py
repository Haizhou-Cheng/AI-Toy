from flask import Flask, request, jsonify, send_file
from paddlespeech.cli.tts import TTSExecutor
import io

app = Flask(__name__)
tts = TTSExecutor(
    task="tts",
    model="fastspeech2_csmsc",
    vocoder="hifigan_csmsc",
    device="gpu"
)

@app.route("/tts", methods=["POST"])
def tts_api():
    text = request.json.get("text", "")
    wav_bytes = tts(text)
    # wav_bytes 是 numpy array，转成二进制
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, wav_bytes, samplerate=tts.sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", as_attachment=False, download_name="out.wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
