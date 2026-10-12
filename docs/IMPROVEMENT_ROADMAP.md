# Improvement Roadmap — FaceVerification

## Version 1.1 — Stabilisation (Do this first)

**Goal:** Make the original application run correctly on Windows.

### Tasks

| # | Task | File | Effort |
|---|---|---|---|
| 1.1.1 | Create `.gitignore` | `.gitignore` | 15 min |
| 1.1.2 | Create `requirements.txt` with pinned versions | `requirements.txt` | 30 min |
| 1.1.3 | Create `.env.example` with REFERENCE_IMAGE_PATH | `.env.example` | 15 min |
| 1.1.4 | Replace headless OpenCV with full GUI build | env | 15 min |
| 1.1.5 | Install pinned dependencies in `.venv` | env | 1–2 hr |
| 1.1.6 | Fix hardcoded path — read from env var or CLI arg | `face-recognition.py:14` | 15 min |
| 1.1.7 | Fix camera failure — sys.exit after error | `face-recognition.py:7` | 5 min |
| 1.1.8 | Validate reference image not None after imread | `face-recognition.py:13` | 10 min |
| 1.1.9 | Broaden exception handling in check_face | `face-recognition.py:21` | 15 min |
| 1.1.10 | Add basic logging (print → logging module) | `face-recognition.py` | 30 min |
| 1.1.11 | Change exit key to q or ESC | `face-recognition.py:39` | 5 min |
| 1.1.12 | Wrap cleanup in try/finally | `face-recognition.py:43` | 10 min |

**Acceptance criteria:**
- Application starts without error
- Camera opens and displays video
- Clear error message if reference image missing
- Clear error message if camera unavailable
- Application closes cleanly on q/ESC

**Dependencies:** Python 3.10, pip, camera hardware

---

## Version 1.2 — Verification Improvements

**Goal:** Make the verification feel responsive and produce stable results.

### Tasks

| # | Task | Effort |
|---|---|---|
| 1.2.1 | Move DeepFace inference to background thread | 3 hr |
| 1.2.2 | Display last cached result while new inference runs | 1 hr |
| 1.2.3 | Smooth result with rolling N-frame majority vote | 2 hr |
| 1.2.4 | Add FPS counter to HUD | 30 min |
| 1.2.5 | Add inference latency display to HUD | 30 min |
| 1.2.6 | Add configurable model via config.yaml | 2 hr |
| 1.2.7 | Add configurable threshold via config.yaml | 1 hr |
| 1.2.8 | Support multiple reference images | 2 hr |
| 1.2.9 | Add face detection bounding box overlay | 1 hr |
| 1.2.10 | Switch to ArcFace model for better accuracy | 1 hr config |

**Acceptance criteria:**
- Video runs at >15 FPS
- Verification result updates every 0.5–2 seconds without freezing display
- Model and threshold are configurable without code changes
- Result is stable (no rapid flickering)

---

## Version 1.3 — Security Improvements

**Goal:** Make the system meaningfully resistant to simple spoofing attacks.

### Tasks

| # | Task | Effort |
|---|---|---|
| 1.3.1 | Research and select passive PAD model | 4 hr |
| 1.3.2 | Integrate liveness check before verification | 8 hr |
| 1.3.3 | Add startup consent notice | 1 hr |
| 1.3.4 | Add verification attempt counter with lockout | 2 hr |
| 1.3.5 | Store reference embeddings (not raw images) | 4 hr |
| 1.3.6 | Add secure file permissions to reference data | 1 hr |
| 1.3.7 | Audit and remove any biometric data from logs | 1 hr |

**Acceptance criteria:**
- Printed photo does not produce MATCH (measured with test set)
- Application locks out after N failed attempts
- No raw face images stored unnecessarily
- User sees consent notice on first run

---

## Version 2.0 — Modern Application

**Goal:** Refactor into a maintainable modular application with proper UX.

### Architecture

```
app/
├── camera/capture.py          # Camera lifecycle (open, read, release)
├── detection/detector.py      # Face detection wrapper
├── verification/verifier.py   # DeepFace wrapper with threading
├── liveness/liveness.py       # PAD model wrapper
├── ui/display.py              # Window, overlays, HUD
└── utils/
    ├── config.py              # Load config.yaml + .env
    └── logger.py              # Structured logging
```

### Tasks

| # | Task | Effort |
|---|---|---|
| 2.0.1 | Create modular directory structure | 1 hr |
| 2.0.2 | Refactor camera logic into CameraCapture class | 2 hr |
| 2.0.3 | Refactor verifier into FaceVerifier class with async | 3 hr |
| 2.0.4 | Create configuration system (YAML + env var) | 2 hr |
| 2.0.5 | Create enrolment workflow (capture + save embedding) | 4 hr |
| 2.0.6 | Add multi-user support | 4 hr |
| 2.0.7 | Modernise UI (status bar, user name display) | 4 hr |
| 2.0.8 | Write unit tests for all modules | 8 hr |
| 2.0.9 | Write integration tests | 4 hr |
| 2.0.10 | Add proper packaging (setup.py or pyproject.toml) | 2 hr |

---

## Version 2.1 — Production Readiness Evaluation

**Goal:** Evaluate and document what would be needed for real deployment.

### Tasks

| # | Task | Effort |
|---|---|---|
| 2.1.1 | Measure FAR, FRR, EER on representative test set | 8 hr |
| 2.1.2 | Measure spoofing resistance with PAD model | 4 hr |
| 2.1.3 | Document remaining production risks | 4 hr |
| 2.1.4 | Security review | 8 hr |
| 2.1.5 | Create installer (PyInstaller or NSIS) | 8 hr |
| 2.1.6 | Write deployment guide | 2 hr |

---

## Recommended First Task

**Implement Version 1.1 — Stabilisation**

Start with task 1.1.1 through 1.1.12.

This can be done in a single session (~4 hours) and will result in a working application for the first time. All subsequent versions depend on this foundation.

Specific first step: create `.gitignore`, `requirements.txt`, `.venv`, and fix the three critical bugs.
