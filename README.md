# Clinical Conversation Transcription System

This application provides speech-to-text transcription with speaker diarization capabilities, specifically designed for clinical conversations between patients and healthcare providers.

## Features

- 🎙️ **Audio Transcription** – Powered by OpenAI's Whisper model  
- 👥 **Speaker Diarization** – Identifies different speakers in conversations  
- 🧑‍⚕️ **Role Classification** – Distinguishes between patients, clinicians, and others  
- 📊 **Structured Output** – JSON format with timestamps and metadata  
- 🌐 **Web Interface** – User-friendly frontend for easy interaction  
- 🐳 **Docker Support** – Containerized deployment for easy setup  

## Prerequisites

- Docker (for containerized deployment)  
- NVIDIA GPU with CUDA support (recommended for optimal performance)  
- At least 16 GB RAM (for Whisper large model)  

## Installation & Setup

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t clinical-transcriber .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 5000:5000 -p 8080:8080 --gpus all clinical-transcriber
   ```

---

### Manual Installation

1. **Install system dependencies:**
   ```bash
   sudo apt-get update && sudo apt-get install -y python3 python3-pip ffmpeg nodejs npm
   ```

2. **Install Python dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Install frontend dependencies:**
   ```bash
   cd FrontEnd
   npm install
   ```

---

## Running the Application

### With Docker
The application automatically starts when the container launches using the `start.sh` script.

### Without Docker
Run both backend and frontend simultaneously:
```bash
./start.sh
```

---

## Accessing the Application

- **Frontend UI:** [http://localhost:8080](http://localhost:8080)  
- **Backend API:** [http://localhost:5000](http://localhost:5000)  

---

## API Documentation

### `POST /transcribe`
Transcribes audio files and identifies speakers.

**Request:**
- Method: `POST`  
- Form-data: `file` (audio file in WAV, MP3, or other common formats)

**Successful Response (200 OK):**
```json
{
  "encounterId": "enc_1620000000",
  "detectedLanguages": ["en"],
  "speakers": [
    {
      "id": "S0",
      "role": "patient",
      "confidence": 0.85
    },
    {
      "id": "S1",
      "role": "clinician",
      "confidence": 0.92
    }
  ],
  "segments": [
    {
      "id": "seg_0001",
      "speaker": "S0",
      "startSec": 1.23,
      "endSec": 4.56,
      "text": "Hello doctor, I've been having headaches",
      "lang": "en"
    },
    {
      "id": "seg_0002",
      "speaker": "S1",
      "startSec": 5.12,
      "endSec": 8.34,
      "text": "How long have you had these symptoms?",
      "lang": "en"
    }
  ],
  "createdAt": "2023-05-15T12:34:56.789Z"
}
```

**Error Responses:**
- `400 Bad Request` – Missing file or invalid request  
- `500 Internal Server Error` – Processing failure  

---

## Project Structure
```
├── app.py                # Backend Flask application
├── Dockerfile            # Docker configuration
├── start.sh              # Launch script
├── requirements.txt      # Python dependencies
└── FrontEnd/             # React frontend
    ├── src/              # React source code
    ├── public/           # Static assets
    ├── package.json      # Frontend dependencies
    └── ...
```

---

## Configuration Options

Edit `app.py` to modify these settings:
```python
# Available models: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL_NAME = "large.en"

# Enable/disable speaker role classification
USE_SPEAKER_ROLE_CLASSIFICATION = True
```

---

## Limitations

- **Speaker Embedding Model** – Current implementation uses a placeholder (replace with ECAPA-TDNN for production)  
- **Resource Intensive** – Whisper large model requires significant RAM (16 GB+ recommended)  
- **Real-time Processing** – Not optimized for real-time transcription  
- **Language Support** – Primarily optimized for English conversations  
- **Speaker Count** – Currently supports up to 3 speakers  

---

## Troubleshooting

**Common Issues:**

- **CUDA Out of Memory:**  
  - Use smaller Whisper model  
  - Reduce audio length  
  - Increase GPU memory  

- **Audio Processing Errors:**  
  - Ensure `ffmpeg` is installed  
  - Verify file is valid audio format  

- **Ollama Not Found:**  
  - Install Ollama or disable speaker role classification  
  - Ensure Ollama is in system PATH  

**Logs Location:**
- **Docker:** `docker logs <container_id>`  
- **Native:** Check terminal output  

---

## Support
For assistance, contact: **mail@mohanish.in**
