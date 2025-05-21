# Sales Call Tone Analysis Backend

A containerized FastAPI service that ingests sales-call audio, runs speaker diarization, Whisper transcription, OpenSMILE feature extraction and LLM-based tone analysis, and returns a consolidated JSON report.

---

## Table of Contents

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Repository Structure](#repository-structure)  
4. [Installation & Setup](#installation--setup)  
   4.1 [Clone the repository](#clone-the-repository)  
   4.2 [Create & activate a Python virtual environment](#create--activate-a-python-virtual-environment)  
   4.3 [Configure environment variables](#configure-environment-variables)  
   4.4 [Install Python dependencies](#install-python-dependencies)  
   4.5 [Run the API locally](#run-the-api-locally)  
   4.6 [Test the upload endpoint](#test-the-upload-endpoint)  
5. [Docker Setup](#docker-setup)  
   5.1 [CPU-only (Local Development)](#cpu-only-local-development)  
   5.2 [GPU-enabled (Production on NVIDIA Server)](#gpu-enabled-production-on-nvidia-server)  
   5.3 [Verify container status & logs](#verify-container-status--logs)  
6. [API Reference](#api-reference)    
7. [Contributing](#contributing)  
8. [License](#license) 
---

## Features

- **Speaker diarization** via PyAnnote  
- **Whisper transcription** (speaker-labeled) via OpenAI API  
- **Acoustic feature extraction** (eGeMAPS) via OpenSMILE  
- **LLM analysis** (pace, pitch, loudness, voice quality, compliance) via `o4-mini`  
- **Dockerized** for CPU or CUDA-enabled GPU deployment  
- **FastAPI** with `/api/upload` endpoint and Swagger UI  

---

## Prerequisites

- **Python 3.8+**  
- **ffmpeg** installed (for `pydub`)  
- **OpenAI API key** (`OPENAI_API_KEY`)  
- **Hugging Face token** for PyAnnote (`PYANNOTE_AUTH_TOKEN`)  
- **Docker** & **Docker Compose** (for containerized deployment)  
- **(GPU only)** NVIDIA driver + CUDA toolkit + NVIDIA Container Toolkit  

---

## Repository Structure

```text
Sales-call-analysis-backend/
├── audio_files/                    # sample audio files
│   ├── test.wav
│   └── …
├── backend/                        # application source
│   ├── convert_audio.py
│   ├── diarized_transcribe.py
│   ├── feature_summarizer.py
│   ├── llm_analyzer.py
│   ├── main.py                     # CLI orchestration
│   ├── open_smile_processor.py
│   └── testing_files/
├── json_transcripts/               # example outputs
├── backend_api_main.py             # FastAPI app entrypoint
├── Dockerfile                      # container build (CPU & GPU logic)
├── docker-compose.yml              # CPU‐only compose
├── docker-compose.gpu.yml          # GPU override compose
├── requirements.txt                # Python deps
├── gpu-requirements.txt            # adds CUDA PyTorch wheels
├── .env                            # API keys (gitignored)
├── .gitignore
└── LICENSE
```

---

## Installation & Setup

Follow these steps to get the Sales Call Tone Analysis backend up and running on your machine (or in Docker).  

---

### 1. Clone the repository

```bash
git clone https://github.com/Jathin04Jan/Sales-call-analysis-backend.git
cd Sales-call-analysis-backend
```

---

### 2. Create & activate a Python virtual environment

**macOS / Linux**  
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**  
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 3. Configure environment variables

Create a file named `.env` in the project root with your API keys:

```ini
OPENAI_API_KEY=sk-<your-openai-key>
PYANNOTE_AUTH_TOKEN=hf_<your-huggingface-token>
```

`.env` is already in `.gitignore`.

---

### 4. Install Python dependencies

**1. Upgrade pip**  
```bash
pip install --upgrade pip
```

**2. Install baseline requirements**  
```bash
pip install -r requirements.txt
```

**3. (Optional – GPU)**  
If you have an NVIDIA GPU + CUDA 12.1, install CUDA‐enabled PyTorch:
```bash
pip install -r gpu-requirements.txt
```

---

### 5. Run the API locally

```bash
uvicorn backend_api_main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

- Server runs at **http://127.0.0.1:8000**  
- Swagger UI: **http://127.0.0.1:8000/docs**

---

### 6. Test the upload endpoint

Using **cURL**:
```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@./audio_files/test.wav" \
  -H "Accept: application/json" | jq .
```

You should see JSON with:
- `call_id`  
- `transcript` (speaker-labeled segments)  
- `features` (OpenSMILE statistics)  
- `summary` (LLM analysis)

---

## Docker Setup

### A) CPU-only (Local Development)

```bash
docker-compose up --build -d
```

- Uses **docker-compose.yml**  
- Builds from `Dockerfile`  
- Runs on CPU inside the container  

### B) GPU-enabled (Production on NVIDIA Server)

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

- Overlays GPU reservation from **docker-compose.gpu.yml**  
- Requires host with NVIDIA drivers + Container Toolkit  

---

### Verify container status & logs

1. **List containers**  
   ```bash
   docker-compose ps
   ```

2. **View logs**  
   ```bash
   docker-compose logs -f backend
   ```

3. **Check GPU availability (production)**  
   ```bash
   docker-compose exec backend python - << 'EOF'
   import torch
   print("CUDA available:", torch.cuda.is_available())
   EOF
   ```

---

Your backend is now fully installed, configured, and containerized—ready for both CPU-only local development and GPU-accelerated production deployments. 

---

## API Reference

### `POST /api/upload`

Submit an audio file to run diarization, transcription, feature extraction and LLM analysis.

- **URL**: `/api/upload`  
- **Method**: `POST`  
- **Headers**:  
  - `Accept: application/json`  
- **Body** (`multipart/form-data`):  
  - **Key**: `file` (type: File) — your `.wav` or `.mp3` audio  

**Example cURL**  
```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@/full/path/to/test.wav" \
  -H "Accept: application/json"
```

**Successful Response** (HTTP 200)  
```json
{
  "call_id": "c7f9e2d4-1234-5678-abcd-9012ef345678",
  "transcript": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.84,
      "end": 5.88,
      "text": "Thank you for the presentation…"
    }
  ],
  "features": {
    "mean_pitch_semitone": 32.7,
    "pitch_stddev_semitone": 5.2,
    "mean_hnr": 8.3,
    "jitter_local": 0.0196
  },
  "summary": "Based on the acoustic features and transcript, the speaker demonstrated a steady mid-range pitch…"
}
```

---

## Contributing

We welcome improvements! Please follow these steps:

1. **Fork** the repository on GitHub  
2. **Clone** your fork & create a feature branch:  
   ```bash
   git clone https://github.com/Jathin04Jan/Sales-call-analysis-backend.git
   cd Sales-call-analysis-backend
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** and add tests if applicable  
4. **Commit** with a clear message:  
   ```bash
   git add .
   git commit -m "Add [short description]"
   ```
5. **Push** your branch:  
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** on the original repo

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  