# FaceVerification v1.3 — Final Development Report

**Date:** 2026-10-12  
**Branch:** feature/faceverification-v1-3  
**Python:** 3.10.0 / Windows 11 Pro

---

## A. Original project

A 44-line prototype that **could not run** on any machine other than the original developer's Mac:
- Reference image hardcoded to `/Users/joshua/Desktop/...`
- DeepFace not installed in the project environment
- `opencv-python-headless` installed (no GUI display)
- No error handling — missing image or camera crash the app immediately
- DeepFace called synchronously on every frame — ~1–3 FPS, UI frozen

---

## B. Version 1.1 — Stabilisation

**Delivered on branch:** `feature/faceverification-modernization`

Fixes implemented:
- Hardcoded path replaced with `REFERENCE_IMAGE_PATH` env var and `--reference` CLI arg
- Camera failure exits cleanly with message (not silent continuation)
- Reference image validated (None check, file existence, readability)
- Exception handling broadened (not just ValueError)
- Inference throttled to every 5 frames (basic responsiveness)
- Full GUI OpenCV installed (`opencv-python==4.10.0.84`)
- DeepFace 0.0.102 installed — no TensorFlow required
- `Q` and `ESC` exit keys
- `try/finally` ensures camera always released
- `requirements.txt`, `.gitignore`, `.env.example` added
- 19 unit tests, all passing

---

## C. Version 1.2 — Performance & Accuracy

**Delivered on branch:** `feature/faceverification-v1-3`

### Async inference
- `InferenceWorker` background thread handles all DeepFace calls
- Main thread (camera + display) never blocked
- Latest-frame-only transfer (no queue buildup)
- Result queue with `maxsize=1` (only latest result kept)
- Controlled shutdown via `threading.Event`

### Frame processing
- Camera display runs independently of inference rate
- Inference interval configurable via `INFERENCE_INTERVAL_MS` (default 500ms)
- FPS counter and inference latency shown in HUD
- Face count displayed

### Model selection
- Default: **ArcFace** with **opencv** detector
- Configurable via `RECOGNITION_MODEL` and `DETECTOR_BACKEND` env vars
- Supported models (via DeepFace): VGG-Face, Facenet, Facenet512, ArcFace, SFace
- Supported detectors: opencv, retinaface, ssd, mtcnn, mediapipe

### Result stability
- `StateManager.tick()` called every display frame
- VERIFIED expires after `RESULT_EXPIRY_MS`; transitions to SCANNING
- No flickering: result only changes when new inference arrives
- Stale VERIFIED never re-used

---

## D. Version 1.3 — Biometric Security

### Anti-spoofing
- `DeepFace.extract_faces(anti_spoofing=True)` used for liveness detection
- Returns `is_real` (bool) and `antispoof_score` (float per face)
- Fail-closed: `is_real=None` or `is_real=False` → SPOOF_DETECTED
- Configurable: `ENABLE_LIVENESS` (default true)

### Verification state machine (10 states)
```
INITIALISING → SCANNING
SCANNING → [inference result]
  → NO_FACE (0 faces)
  → MULTIPLE_FACES (>1 faces)
  → SPOOF_DETECTED (liveness check failed)
  → VERIFIED (identity match + liveness pass)
  → REJECTED (identity mismatch)
  → ERROR (any exception)
VERIFIED → SCANNING (after RESULT_EXPIRY_MS)
```

### Security guarantees (tested)
- Only `VERIFIED` + freshness → `is_authenticated()` == True
- All 9 other states → `is_authenticated()` == False
- Any exception in inference pipeline → ERROR (never VERIFIED)
- VERIFIED expires; expired VERIFIED → not authenticated
- Multiple faces → no authentication
- Zero faces → no authentication

---

## E. Architecture

```
face_verification.py (main loop + CLI)
    │
    ├── app/config.py          AppConfig dataclass
    ├── app/camera.py          CameraManager context manager
    ├── app/state_manager.py   VerificationState, VerificationResult, StateManager
    ├── app/inference_worker.py  Background thread (detect→liveness→verify)
    ├── app/overlay.py         HUD drawing
    └── app/logging_config.py  Logging setup
```

See `docs/ARCHITECTURE.md` for full module and threading documentation.

---

## F. Machine learning

| Component | Implementation | Notes |
|---|---|---|
| Face detector | OpenCV Haar cascade (default) | Fast, lower accuracy than RetinaFace |
| Recognition model | ArcFace | Downloaded on first run (~137 MB) |
| Anti-spoofing | DeepFace built-in PAD model | Shallow CNN, RGB-only |
| Backend | DeepFace 0.0.102 (ONNX/native) | No TensorFlow required |

**ArcFace selection rationale:** DeepFace documentation cites ArcFace as one of the highest-accuracy models (~99.4% on LFW). No independent benchmark was performed on this hardware. The model is configurable.

---

## G. Performance

Actual runtime measurements require a physical webcam. The following are architecture-level expectations based on the implementation:

| Metric | Expected | Notes |
|---|---|---|
| Display FPS | 20–30 FPS | Camera capture not blocked by inference |
| Inference latency | 0.3–2.0 s | CPU-dependent; ArcFace is heavier than SFace |
| Inference rate | ~1–2/s | Controlled by `INFERENCE_INTERVAL_MS` |
| Worker thread count | 1 | Single background thread |

*No fabricated benchmark numbers. Real measurements require webcam and reference image.*

---

## H. Security

See `docs/SECURITY_ASSESSMENT.md` for full assessment.

**Summary:**
- Fail-closed state machine: ✓
- Anti-spoofing (passive PAD): ✓ (partial — RGB camera limitations apply)
- Multiple-face protection: ✓
- Result expiry: ✓
- No biometric data stored: ✓
- Virtual camera injection: ✗ Not mitigated
- 3D mask: ✗ Not mitigated

---

## I. Testing

```
75 tests, 75 passed, 0 failed — 2.07s
```

| Test file | Count | Coverage |
|---|---|---|
| test_config.py | 16 | Config loading, defaults, CLI overrides |
| test_face_verification.py | 19 | v1.1 compat: image loading, camera, verify_face, overlay |
| test_inference_worker.py | 9 | Lifecycle, frame handling, error paths |
| test_security.py | 18 | Fail-closed: all non-VERIFIED states, expiry |
| test_state_manager.py | 13 | Expiry transitions, is_authenticated |

**Not unit-tested (requires hardware):**
- Live webcam frame flow
- Actual DeepFace model inference
- Real anti-spoofing with printed photos

---

## J. Setup (Windows)

```powershell
git clone https://github.com/TechieTojin/FaceVerification.git
cd FaceVerification
git checkout feature/faceverification-v1-3

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# Edit .env: set REFERENCE_IMAGE_PATH=C:\path\to\your_face.jpg

python face_verification.py
# or:
python face_verification.py --reference "C:\path\to\photo.jpg" --camera 0
```

First run: ArcFace model (~137 MB) downloads automatically.

---

## K. Remaining work before production use

1. **Independent anti-spoofing evaluation** — test PAD model against printed photos and screen replays; measure FAR/FRR
2. **Camera authenticity** — no defence against virtual camera injection
3. **Threshold calibration** — use representative evaluation data to calibrate ArcFace threshold for the target population
4. **Attempt limiting** — `MAX_VERIFICATION_ATTEMPTS` config exists but lockout not yet implemented
5. **Reference embedding encryption** — store encrypted embeddings rather than raw images
6. **Multi-user enrolment** — current system supports one reference image per session
7. **Demographic evaluation** — test accuracy across representative demographics
8. **Legal review** — biometric data regulations vary by jurisdiction
