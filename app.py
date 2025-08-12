# app.py
import os
import json
import tempfile
import time
import subprocess
import shlex
import shutil
from datetime import datetime
from flask import Flask, request, jsonify
import whisper
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
import torchaudio
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# ------------- Configuration ----------------
WHISPER_MODEL_NAME = "large.en"   # change to "small"/"medium"/"large" if you want
# ------------------------------------------------

# Load Whisper model once (may take RAM)
logging.info("Loading Whisper model...")
WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
logging.info("Whisper loaded.")

class SpeakerEmbeddingModel:
    """Placeholder embedding model. Replace with a real speaker embedding model in production."""
    def __init__(self):
        self.dim = 256

    def embed(self, audio_np: np.ndarray, sample_rate: int):
        """
        audio_np: 1D numpy array (samples,)
        returns: 1D numpy vector
        """
        # NOTE: Replace with a real speaker embedding model like ECAPA-TDNN
        # Return deterministic-ish vector for reproducibility during testing
        rng = np.random.RandomState(abs(int(np.sum(audio_np))) % (2**31 - 1))
        return rng.randn(self.dim).astype(np.float32)

EMBEDDING_MODEL = SpeakerEmbeddingModel()


def convert_to_wav(input_path, output_path):
    """Convert audio file to 16kHz mono WAV using pydub (ffmpeg required)."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")


def run_ollama(prompt, model="medllama2"):
    """Try to run Ollama if installed; otherwise return 'other' as fallback."""
    if shutil.which("ollama") is None:
        logging.warning("Ollama not found on PATH; skipping Ollama.")
        return "other"
    try:
        # Use shlex.quote to avoid shell-injection issues
        quoted = shlex.quote(prompt)
        cmd = f"ollama run {shlex.quote(model)} {quoted}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=20
        )
        out = result.stdout.strip()
        return out if out else "other"
    except Exception as e:
        logging.exception("Ollama error")
        return "other"


def classify_speaker_role(transcript):
    """
    Try to classify speaker role using Ollama, but fall back to a simple heuristic.
    Returns one of: 'patient', 'clinician', 'other'
    """
    # heuristic fallback
    text = (transcript or "").lower()
    # quick keyword heuristics
    if any(tok in text for tok in ["i have", "my pain", "i am feeling", "i've been", "i'm having", "my symptoms"]):
        heuristic = "patient"
    elif any(tok in text for tok in ["doctor", "dr.", "dr ", "prescribe", "examination", "diagnos", "recommend", "order", "reviewing"]):
        heuristic = "clinician"
    else:
        heuristic = "other"

    # Try Ollama, but keep heuristic if Ollama is missing/returns invalid.
    response = run_ollama(transcript)
    response = (response or "").strip().lower()
    if response in ("patient", "clinician", "other"):
        return response
    return heuristic


def frame_rms(waveform: torch.Tensor, sample_rate: int, frame_ms=30, hop_ms=10):
    """
    Compute RMS energy per frame.
    waveform: torch tensor shape (1, n_samples)
    returns: numpy array of frame RMS values
    """
    frame_len = int(frame_ms * sample_rate / 1000)
    hop_len = int(hop_ms * sample_rate / 1000)
    if frame_len <= 0: frame_len = 1
    if hop_len <= 0: hop_len = 1
    # pad end so last frame included
    n = waveform.shape[1]
    pad = (frame_len - (n - frame_len) % hop_len) % hop_len
    x = torch.nn.functional.pad(waveform, (0, pad))
    frames = x.unfold(1, frame_len, hop_len)  # shape (1, n_frames, frame_len)
    rms = torch.sqrt((frames ** 2).mean(dim=2) + 1e-8)
    return rms.squeeze(0).cpu().numpy(), frame_len, hop_len


def diarize_audio(audio_path):
    """
    Lightweight diarization:
      - compute frame-based VAD (RMS)
      - extract contiguous speech regions longer than min_duration
      - compute a simple embedding for each region and cluster them
    Returns list of {"start": float, "end": float, "speaker": "S0"} dictionaries
    """
    waveform, sample_rate = torchaudio.load(audio_path)  # shape (channels, samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.contiguous()

    # compute RMS per frame
    rms, frame_len, hop_len = frame_rms(waveform, sample_rate, frame_ms=30, hop_ms=10)
    # VAD threshold: adaptive
    thresh = max(1e-4, np.median(rms) * 1.5)
    speech_frames = rms > thresh

    # find contiguous speech frames and convert to time
    segments = []
    in_seg = False
    seg_start_frame = 0
    for i, s in enumerate(speech_frames):
        if s and not in_seg:
            in_seg = True
            seg_start_frame = i
        elif not s and in_seg:
            in_seg = False
            start_sec = seg_start_frame * hop_len / sample_rate
            end_sec = (i * hop_len + frame_len) / sample_rate
            if end_sec - start_sec >= 0.25:  # min 250ms
                segments.append((start_sec, end_sec))
    if in_seg:
        start_sec = seg_start_frame * hop_len / sample_rate
        end_sec = (len(speech_frames) * hop_len + frame_len) / sample_rate
        if end_sec - start_sec >= 0.25:
            segments.append((start_sec, end_sec))

    # Extract embeddings
    embeddings = []
    valid_segments = []
    for (s, e) in segments:
        s_sample = int(max(0, s * sample_rate))
        e_sample = int(min(waveform.shape[1], e * sample_rate))
        if e_sample - s_sample < 200:  # skip too short
            continue
        seg_audio = waveform[0, s_sample:e_sample].cpu().numpy()
        emb = EMBEDDING_MODEL.embed(seg_audio, sample_rate)  # 1D vector
        if emb is None:
            continue
        embeddings.append(emb)
        valid_segments.append((s, e))

    # If no embeddings found, return empty to let caller fallback
    if not embeddings:
        return []

    embeddings = np.vstack(embeddings)  # (n_segs, dim)
    n_clusters = min(3, len(embeddings))  # assume up to 3 speakers
    if len(embeddings) == 1:
        labels = [0]
    else:
        # AgglomerativeClustering with cosine affinity; sklearn may warn if metric unsupported,
        # but many versions accept affinity='cosine' or metric='cosine' depending on version.
        try:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
            labels = clustering.fit_predict(embeddings)
        except TypeError:
            # fallback to Euclidean if affinity not supported
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
            labels = clustering.fit_predict(embeddings)

    diarization = []
    for (seg, lbl) in zip(valid_segments, labels):
        diarization.append({
            "start": seg[0],
            "end": seg[1],
            "speaker": f"S{int(lbl)}"
        })
    return diarization


def process_audio(audio_path):
    """Process audio path -> produce structured JSON output"""
    diarization = diarize_audio(audio_path)

    # Transcribe with Whisper (segment-level timestamps)
    # note: whisper.transcribe returns dict with 'segments' list each having text, start, end
    result = WHISPER_MODEL.transcribe(audio_path)

    # Build a list of words if whisper provides segments with words; otherwise use segments as blocks
    words = []
    # Whisper by default doesn't give word-level timestamps. If it does, code can be adapted.
    for seg in result.get("segments", []):
        # Try to split segment text into words and assign approximate timestamps by linear interpolation.
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        tokens = seg_text.split()
        if len(tokens) == 1:
            words.append({"start": start, "end": end, "text": tokens[0]})
        else:
            # distribute times evenly across tokens (approximation)
            dur = max(1e-6, end - start)
            per = dur / len(tokens)
            for i, w in enumerate(tokens):
                words.append({
                    "start": start + i * per,
                    "end": start + (i + 1) * per,
                    "text": w
                })

    # If no diarization, fallback: single speaker for whole transcript
    if not diarization:
        full_text = result.get("text", "").strip()
        seg_start = result.get("segments", [{}])[0].get("start", 0.0) if result.get("segments") else 0.0
        seg_end = result.get("segments", [{}])[-1].get("end", seg_start) if result.get("segments") else seg_start
        speaker_id = "S0"
        role = classify_speaker_role(full_text)
        speakers = [{"id": speaker_id, "role": role, "confidence": 0.8}]
        segments = [{
            "id": "seg_0000",
            "speaker": speaker_id,
            "startSec": round(seg_start, 2),
            "endSec": round(seg_end, 2),
            "text": full_text,
            "lang": "en"
        }]
        return {
            "encounterId": f"enc_{int(time.time())}",
            "detectedLanguages": ["en"],
            "speakers": speakers,
            "segments": segments,
            "createdAt": datetime.utcnow().isoformat() + "Z"
        }

    # Assign words to diarized segments
    segments_out = []
    speaker_roles = {}
    speaker_counter = {}
    for i, dseg in enumerate(diarization):
        s = dseg["start"]
        e = dseg["end"]
        seg_words = [w for w in words if w["start"] >= s - 1e-3 and w["end"] <= e + 1e-3]
        if not seg_words:
            # If words didn't line up, approximate by taking transcripts that overlap
            overlap_words = [w for w in words if not (w["end"] < s or w["start"] > e)]
            seg_words = overlap_words

        if not seg_words:
            text = ""
        else:
            text = " ".join(w["text"] for w in seg_words).strip()

        speaker_id = dseg["speaker"]
        if speaker_id not in speaker_roles:
            role = classify_speaker_role(text)
            speaker_roles[speaker_id] = role
            speaker_counter[speaker_id] = 1
        else:
            speaker_counter[speaker_id] += 1

        segments_out.append({
            "id": f"seg_{i:04d}",
            "speaker": speaker_id,
            "startSec": round(s, 2),
            "endSec": round(e, 2),
            "text": text,
            "lang": "en"
        })

    speakers = [{
        "id": spk,
        "role": role,
        "confidence": min(0.99, 0.7 + 0.1 * min(5, speaker_counter[spk]))
    } for spk, role in speaker_roles.items()]

    return {
        "encounterId": f"enc_{int(time.time())}",
        "detectedLanguages": ["en"],
        "speakers": speakers,
        "segments": segments_out,
        "createdAt": datetime.utcnow().isoformat() + "Z"
    }


@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    tmp_path = None
    wav_path = None
    try:
        # Save uploaded file to a temp filename with original extension
        _, ext = os.path.splitext(audio_file.filename)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        # Convert to wav if not wav already
        if not audio_file.filename.lower().endswith('.wav'):
            wav_path = tmp_path + ".wav"
            convert_to_wav(tmp_path, wav_path)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            audio_path = wav_path
        else:
            audio_path = tmp_path

        result = process_audio(audio_path)

        return jsonify(result)

    except Exception as e:
        logging.exception("Processing error")
        return jsonify({"error": str(e)}), 500

    finally:
        for p in (tmp_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


if __name__ == "__main__":
    # For development only. Use gunicorn/uvicorn in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
