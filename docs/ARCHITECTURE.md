# Architecture — FaceVerification v1.3

## Overview

The application is a modular Python desktop program built around three concurrent concerns:

1. **Camera capture / display loop** — runs in the main thread at target frame rate
2. **Inference worker** — runs in a single background thread; never blocks the display
3. **State machine** — holds the current `VerificationResult` and applies expiry rules

---

## Directory structure

```
FaceVerification/
├── app/
│   ├── __init__.py          — package exports (VerificationState, VerificationResult, AppConfig)
│   ├── config.py            — AppConfig dataclass loaded from .env / env vars
│   ├── camera.py            — CameraManager context manager
│   ├── state_manager.py     — VerificationState enum, VerificationResult, StateManager
│   ├── inference_worker.py  — InferenceWorker background thread (full detection pipeline)
│   ├── overlay.py           — draw_overlay() — HUD drawing on frames
│   └── logging_config.py    — setup_logging()
├── tests/
│   ├── test_config.py
│   ├── test_face_verification.py   — v1.1 compat tests
│   ├── test_inference_worker.py
│   ├── test_security.py
│   └── test_state_manager.py
├── face_verification.py     — entry point (main loop, CLI, v1.1 shims)
├── face-recognition.py      — original 44-line prototype (preserved)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Threading model

```
Main thread                         InferenceWorker thread
──────────────────────────────      ──────────────────────────────
cap.read() → frame                  wait for frame_event
worker.push_frame(frame)     ─────▶ _get_frame() → latest frame
worker.result_queue.get_nowait()    _process(frame):
state_mgr.update(result)      ◀─────  extract_faces (liveness)
state_mgr.tick()                       count faces
draw_overlay(frame, result)            verify identity
cv2.imshow()                    ─────▶ result_queue.put_nowait()
cv2.waitKey(1)
```

**Key design choices:**
- Only the latest frame is stored — old frames are discarded, not queued
- Output queue has `maxsize=1` — only latest result is kept
- The main thread is never blocked by DeepFace inference
- Worker is a daemon thread — cleaned up automatically if main thread crashes

---

## Verification state machine

```
INITIALISING ──▶ SCANNING ──────────────────────────────────────────▶ SCANNING (expiry reset)
                    │                                                      ▲
                    ▼                                                      │ expiry
              [frame arrives]                                         VERIFIED ──▶ SCANNING
                    │
              [InferenceWorker._process()]
                    │
              ┌─────┴──────────┐
              │ face_count=0   │ face_count>1   │ exception
              ▼                ▼                ▼
           NO_FACE    MULTIPLE_FACES          ERROR
                    │
              [liveness check — ENABLE_LIVENESS=true]
              is_real=None or False
                    ▼
             SPOOF_DETECTED
                    │
              [DeepFace.verify()]
              ┌─────┴─────┐
          verified=True  verified=False   exception
              ▼                ▼               ▼
           VERIFIED        REJECTED          ERROR
```

**Fail-closed rule:** Any state that is not `VERIFIED` means NOT authenticated.
`is_authenticated()` additionally checks that VERIFIED has not expired.

---

## module: app/config.py

`AppConfig` is a Python `@dataclass` populated via `AppConfig.from_env()`.

| Field | Env var | Default |
|---|---|---|
| `reference_image_path` | `REFERENCE_IMAGE_PATH` | `""` |
| `camera_index` | `CAMERA_INDEX` | `0` |
| `camera_width` | `CAMERA_WIDTH` | `640` |
| `camera_height` | `CAMERA_HEIGHT` | `480` |
| `recognition_model` | `RECOGNITION_MODEL` | `"ArcFace"` |
| `detector_backend` | `DETECTOR_BACKEND` | `"opencv"` |
| `enable_liveness` | `ENABLE_LIVENESS` | `True` |
| `enable_performance_overlay` | `ENABLE_PERFORMANCE_OVERLAY` | `True` |
| `inference_interval_ms` | `INFERENCE_INTERVAL_MS` | `500.0` |
| `result_expiry_ms` | `RESULT_EXPIRY_MS` | `1500.0` |
| `max_verification_attempts` | `MAX_VERIFICATION_ATTEMPTS` | `10` |
| `log_level` | `LOG_LEVEL` | `"INFO"` |

---

## module: app/inference_worker.py — detection pipeline

Each call to `_process(frame)` runs:

1. **Frame validation** — null/empty frame → ERROR
2. **Face extraction** — `DeepFace.extract_faces(anti_spoofing=True)` when liveness enabled
3. **Face count** — 0 → NO_FACE, >1 → MULTIPLE_FACES
4. **Liveness check** — `face[0]['is_real']` must be `True` → else SPOOF_DETECTED
5. **Identity verification** — `DeepFace.verify(frame, reference, model_name, detector_backend)`
6. **Decision** — `verified=True` → VERIFIED, `False` → REJECTED
7. **Any exception at any step** → ERROR (never VERIFIED)

Every result includes `timestamp`, `inference_ms`, `distance`, `threshold`, `liveness_score`, `face_count`.

---

## module: app/state_manager.py — expiry

`StateManager.tick()` is called every display frame.

If the current state is `VERIFIED` and `time.monotonic() - result.timestamp >= expiry_s`:
- Transitions to `SCANNING`
- The VERIFIED authentication window has closed

`is_authenticated()` checks both `state == VERIFIED` AND freshness simultaneously — no TOCTOU gap within the same thread.
