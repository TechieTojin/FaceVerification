# Issues and Risks — FaceVerification

## Critical Bugs (application cannot run)

### BUG-01: Hardcoded macOS reference image path
- **File:** `face-recognition.py:14`
- **Code:** `"/Users/joshua/Desktop/Personal/referenceimage2.jpg"`
- **Root cause:** Absolute path committed from the original developer's Mac. Will never resolve on any other machine.
- **Impact:** `reference_img = None` → crash on first frame with `AttributeError`.
- **Fix:** Accept path via CLI argument or environment variable. Provide `.env.example`.

### BUG-02: Camera failure does not stop execution
- **File:** `face-recognition.py:6–7`
- **Code:** `if not cap.isOpened(): print("Error...")`
- **Root cause:** Error is printed but execution continues.
- **Impact:** `cap.read()` returns `(False, None)` every iteration; application loops indefinitely with no output.
- **Fix:** `sys.exit(1)` after the error print.

### BUG-03: NoneType crash on missing reference image
- **File:** `face-recognition.py:19`
- **Code:** `reference_img.copy()`
- **Root cause:** `cv2.imread` returns `None` silently; `.copy()` raises `AttributeError`.
- **Impact:** App crashes on first frame with unhandled exception.
- **Fix:** Validate `reference_img is not None` after imread and exit with a clear message.

### BUG-04: DeepFace not installed
- **Environment:** Global Python 3.10.0
- **Installed:** numpy 2.2.6, pillow 12.1.1, opencv-python-headless 5.0.0.93
- **Missing:** deepface, tensorflow, keras
- **Impact:** `from deepface import DeepFace` fails with `ModuleNotFoundError` — app cannot start.
- **Fix:** See SETUP_GUIDE.md for installation instructions.

### BUG-05: opencv-python-headless installed — no GUI support
- **Package:** `opencv-python-headless 5.0.0.93`
- **Impact:** `cv2.imshow` raises `cv2.error` or silently fails — no video window appears.
- **Fix:** Uninstall headless variant, install `opencv-python==4.10.0.84`.

---

## High Severity Issues

### ISS-01: DeepFace called on every frame (no throttle)
- **File:** `face-recognition.py:28`
- **Impact:** Inference takes 0.3–2s per call on CPU. Effective frame rate: ~1–3 FPS. Window freezes between frames.
- **Fix:** Run inference in a background thread. Cache result and display it until the next inference completes.

### ISS-02: Only ValueError caught in check_face
- **File:** `face-recognition.py:21`
- **Impact:** AttributeError (missing image), OSError (file system), TensorFlow errors, and model errors all propagate uncaught.
- **Fix:** Catch `Exception` broadly with structured logging, or catch specific expected exceptions.

### ISS-03: No liveness detection
- **Impact:** A photograph or video of the reference person held in front of the camera will produce a MATCH. The system is trivially spoofable.
- **Fix:** Integrate a passive anti-spoofing model (e.g., Silent-Face-Anti-Spoofing or similar).

### ISS-04: numpy 2.x incompatible with TensorFlow 2.15
- **Environment:** numpy 2.2.6 is installed. TensorFlow 2.15 requires numpy < 2.0.
- **Impact:** TensorFlow will fail to import with a binary compatibility error.
- **Fix:** Pin numpy==1.26.4 in requirements.txt and reinstall.

---

## Medium Severity Issues

### ISS-05: No .gitignore
- **Risk:** Virtual environment, model weights (~500 MB), reference images, and .env files could be accidentally committed.

### ISS-06: No requirements.txt
- **Risk:** No reproducible environment. Any other developer will face the same dependency problems.

### ISS-07: No error handling for camera disconnect
- **Impact:** If camera is unplugged during operation, `cap.read()` returns `(False, None)` and the loop continues silently.

### ISS-08: Non-standard exit key (j)
- **Impact:** Users expect `q` or ESC. `j` is undocumented and unintuitive.

### ISS-09: No logging
- **Impact:** Failures are silent. When something goes wrong, there is no record of what happened or why.

### ISS-10: Global variables (cap, reference_img)
- **Impact:** Makes testing and refactoring difficult. State cannot be injected or mocked.

---

## Low Severity Issues

### ISS-11: No type hints or docstrings
- **Impact:** Reduced readability and IDE support.

### ISS-12: Undefined behaviour with multiple faces
- **Impact:** DeepFace uses the largest/first detected face — undocumented assumption.

### ISS-13: No FPS display
- **Impact:** User has no feedback about whether the system is running or frozen.

### ISS-14: Camera resolution setting not verified
- **Impact:** Camera may silently ignore the 640×480 request.

---

## Security Risks

| Risk | Severity | Description |
|---|---|---|
| Photo/video spoofing | Critical | No liveness — printed photo will match |
| No attempt limiting | High | Unlimited verification attempts |
| Reference image unprotected | High | Plain filesystem access, no encryption |
| Developer name in path | Medium | `/Users/joshua/...` leaks personal information |
| No consent mechanism | High | Biometric processing with no user awareness |
| No data minimisation | Medium | Frames processed without need to persist — acceptable, but embeddings should never be logged |

---

## Technical Debt

| Item | Effort to fix |
|---|---|
| No virtual environment or requirements.txt | 30 min |
| Hardcoded path | 15 min |
| Missing exception handling | 1 hr |
| Blocking inference in main loop | 2–3 hr |
| No tests | 4–8 hr |
| No configuration management | 2 hr |
| No liveness detection | 8–16 hr |
