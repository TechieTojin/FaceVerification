# FaceVerification

Real-time face verification using a webcam and a reference image. Displays **MATCH** or **NOT A MATCH** on the live video feed.

---

## Quick start (Windows)

### 1. Clone and enter the project

```powershell
git clone https://github.com/TechieTojin/FaceVerification.git
cd FaceVerification
git checkout feature/faceverification-modernization
```

### 2. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **First run:** ArcFace model weights (~137 MB) are downloaded automatically from the internet.

### 3. Configure the reference image

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set the path to a clear, well-lit photo of the person to verify:

```
REFERENCE_IMAGE_PATH=C:\Users\YourName\Pictures\your_face.jpg
```

### 4. Run

```powershell
python face_verification.py
```

Or pass the reference image directly:

```powershell
python face_verification.py --reference "C:\path\to\photo.jpg"
```

Press **Q** or **ESC** in the video window to quit.

---

## Configuration

All settings can be set in `.env` or passed as CLI arguments.

| Environment variable | CLI flag | Default | Description |
|---|---|---|---|
| `REFERENCE_IMAGE_PATH` | `--reference` / `-r` | *(required)* | Path to the reference face image |
| `CAMERA_INDEX` | `--camera` / `-c` | `0` | Camera device index |
| `FACE_MODEL` | — | `ArcFace` | DeepFace model (`ArcFace`, `VGG-Face`, `Facenet512`, `SFace`) |
| `DETECTOR_BACKEND` | — | `opencv` | Face detector (`opencv`, `retinaface`, `ssd`) |

---

## Run tests

```powershell
.\.venv\Scripts\pytest tests/ -v
```

---

## Project files

| File | Purpose |
|---|---|
| `face_verification.py` | Main application (v1.1) |
| `face-recognition.py` | Original prototype (preserved) |
| `requirements.txt` | Pinned dependencies |
| `.env.example` | Configuration template |
| `tests/` | Unit tests (pytest) |
| `docs/` | Technical documentation |

---

## Technology

- **OpenCV 4.10** — webcam capture and display
- **DeepFace 0.0.102** — face verification (ArcFace model by default)
- **python-dotenv** — `.env` configuration loading

---

## Limitations (v1.1)

- No liveness detection — a printed photo of the reference person may produce a match
- Inference runs every 5 frames; the display updates every frame
- Single reference image per session
- No multi-user support

See [`docs/IMPROVEMENT_ROADMAP.md`](docs/IMPROVEMENT_ROADMAP.md) for the planned improvements.
