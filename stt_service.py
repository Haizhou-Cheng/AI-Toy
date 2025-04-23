from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)
model = WhisperModel(
    model_size_or_path="large-v2",
    device="cuda",
    compute_type="float16"
)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files['file'].stream.read()
    # 假设上传的是 WAV PCM 16k 单声道
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="Chinese"
    )
    text = "".join([seg.text for seg in segments])
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
