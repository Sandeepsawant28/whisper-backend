"""
Whisper Small Konkani (CTranslate2) Model Server — Hugging Face Space deploy
Loads the fine-tuned model from the HF Hub at runtime (no local model files
in this repo). Serves both a REST /transcribe endpoint and live streaming
transcription over Socket.IO.
"""
import os
import gc
import wave
import tempfile
import traceback
import threading
import subprocess

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

HF_REPO_ID = "sandeepsawant28/whisper-small-konkani-numbers"

DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

model = None
transcribe_lock = threading.Lock()

STREAM_MAX_BUFFER_SECONDS = 12
STREAM_MIN_SECONDS = 1.0


def load_model():
    global model, DEVICE, COMPUTE_TYPE
    try:
        print(f"Downloading model from {HF_REPO_ID}...")
        # HF_TOKEN env var (set as a Space secret) is used automatically by
        # snapshot_download if the repo is private; no need to pass it manually.
        local_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=["ct2/*"])
        model_path = os.path.join(local_path, "ct2")
        print(f"Loading CTranslate2 model from {model_path} on {DEVICE} ({COMPUTE_TYPE})...")
        model = WhisperModel(model_path, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"[OK] Model loaded successfully on {DEVICE.upper()} ({COMPUTE_TYPE}).")
    except Exception as err:
        print(f"Notice: Model load failed ({err}).")
        traceback.print_exc()
        model = None


try:
    load_model()
except Exception as e:
    print(f"Notice: Model load failed ({e}).")
    traceback.print_exc()
    model = None

app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                     max_http_buffer_size=20_000_000)
streaming_sessions = {}


def convert_to_wav(webm_path, wav_path):
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", webm_path,
            "-ar", "16000", "-ac", "1",
            "-f", "wav", wav_path
        ],
        capture_output=True,
        timeout=15
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore"))


def wav_to_float32(wav_path):
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    with transcribe_lock:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
            audio_file.save(temp_webm.name)
            webm_path = temp_webm.name

        wav_path = webm_path.replace(".webm", ".wav")
        segments, info = None, None

        try:
            if os.path.getsize(webm_path) < 600:
                return jsonify({"status": "empty", "text": "", "language": "Konkani (kok)"})

            try:
                convert_to_wav(webm_path, wav_path)
            except Exception as ffmpeg_err:
                return jsonify({
                    "status": "warning",
                    "text": "",
                    "warning": f"Audio decoding failed: {ffmpeg_err}"
                }), 200

            if model is None:
                return jsonify({
                    "status": "warning",
                    "is_mock": True,
                    "error": "Model not loaded on server.",
                    "language": "Konkani (kok)"
                }), 503

            segments, info = model.transcribe(
                wav_path,
                language="mr",
                beam_size=1,
                best_of=1,
                vad_filter=False,
            )
            text = " ".join([seg.text for seg in segments]).strip()

            if not text:
                return jsonify({"status": "empty", "text": "", "language": "Konkani (kok)"})

            return jsonify({
                "status": "success",
                "is_mock": False,
                "text": text,
                "language": "Konkani (kok)",
                "device": DEVICE,
            })

        except Exception as err:
            traceback.print_exc()
            return jsonify({"error": str(err)}), 500
        finally:
            for p in (webm_path, wav_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            try:
                del segments, info
            except NameError:
                pass
            gc.collect()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online" if model is not None else "model_not_loaded",
        "model": "Whisper Small Konkani Numbers (CTranslate2)",
        "backend": "faster-whisper / CTranslate2",
        "hardware": DEVICE.upper(),
        "compute_type": COMPUTE_TYPE,
    })


@socketio.on("connect")
def on_stream_connect():
    sid = request.sid
    streaming_sessions[sid] = bytearray()
    print(f"[stream] client connected: {sid}")


@socketio.on("disconnect")
def on_stream_disconnect():
    sid = request.sid
    streaming_sessions.pop(sid, None)
    print(f"[stream] client disconnected: {sid}")


@socketio.on("stop_stream")
def on_stop_stream():
    sid = request.sid
    streaming_sessions[sid] = bytearray()


@socketio.on("audio_chunk")
def on_audio_chunk(data):
    sid = request.sid
    if sid not in streaming_sessions:
        streaming_sessions[sid] = bytearray()

    streaming_sessions[sid].extend(data)
    webm_bytes = bytes(streaming_sessions[sid])

    if len(webm_bytes) < 600:
        return

    webm_path = wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
            temp_webm.write(webm_bytes)
            webm_path = temp_webm.name
        wav_path = webm_path.replace(".webm", ".wav")

        try:
            convert_to_wav(webm_path, wav_path)
        except Exception:
            return

        audio = wav_to_float32(wav_path)

        max_samples = STREAM_MAX_BUFFER_SECONDS * 16000
        if len(audio) > max_samples:
            audio = audio[-max_samples:]

        if len(audio) < int(STREAM_MIN_SECONDS * 16000):
            return

        if model is None:
            return

        with transcribe_lock:
            segments, info = model.transcribe(
                audio, language="mr", beam_size=1, best_of=1, vad_filter=False
            )
            text = " ".join(seg.text for seg in segments).strip()

        socketio.emit("partial_transcript", {"text": text, "language": "Konkani (kok)"}, room=sid)

    except Exception:
        traceback.print_exc()
    finally:
        for p in (webm_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        gc.collect()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    print(f"\n[RUNNING] Whisper Konkani API running on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
