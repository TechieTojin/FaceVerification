# Code Walkthrough — face-recognition.py

## Overview

The entire application is 44 lines in a single flat script. There are no classes, no modules, and no configuration files. Below is a step-by-step explanation of every section.

---

## Line-by-line walkthrough

### Lines 1–2: Imports

```python
import cv2
from deepface import DeepFace
```

- `cv2` — OpenCV. Used for opening the webcam, reading frames, drawing text overlays, and displaying the video window.
- `DeepFace` — a high-level face recognition library that wraps multiple deep learning models (VGG-Face, FaceNet, ArcFace, etc.). The `verify` method takes two images and returns whether they contain the same person.

**What happens at import time:** When `from deepface import DeepFace` executes, DeepFace imports TensorFlow internally. If TensorFlow is not installed, the application crashes here with `ModuleNotFoundError`.

---

### Lines 5–10: Camera initialisation

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

- `cv2.VideoCapture(0)` — opens the default camera (index 0). On most systems this is the built-in webcam.
- `cap.isOpened()` — returns True if the camera was successfully opened. The error print here is misleading: execution continues even if the camera failed. This is a bug.
- `cap.set(...)` — requests 640×480 resolution. The camera driver may ignore this and use its own default.

---

### Lines 13–14: Reference image loading

```python
reference_img = cv2.imread(
    "/Users/joshua/Desktop/Personal/referenceimage2.jpg")
```

- `cv2.imread` — reads an image file from disk into a numpy array (BGR format).
- The path `/Users/joshua/Desktop/Personal/referenceimage2.jpg` is an absolute macOS path. **It does not exist on Windows.** OpenCV will silently return `None` without raising any exception.
- After this line, `reference_img` is `None` on any machine other than the original developer's Mac.

---

### Lines 17–22: check_face function

```python
def check_face(frame):
    try:
        result = DeepFace.verify(frame, reference_img.copy())
        return result['verified']
    except ValueError:
        return False
```

This is the only function in the application.

**Parameters:**
- `frame` — a numpy array representing one video frame (shape: 480×640×3, dtype uint8, BGR colour order)

**Returns:**
- `True` if the faces match
- `False` if no face is detected, or verification fails

**What DeepFace.verify does internally:**
1. Detects faces in both images using the configured detector backend (default: OpenCV Haar cascade).
2. Aligns and crops each detected face.
3. Passes each cropped face through the VGG-Face neural network to produce a 4096-dimensional vector (embedding).
4. Computes cosine distance between the two embeddings.
5. Compares the distance to the configured threshold (default: 0.40).
6. Returns a dict containing `{'verified': bool, 'distance': float, 'threshold': float, ...}`.

**Bugs in this function:**
- `reference_img.copy()` — if `reference_img` is `None` (missing file), this raises `AttributeError`, which is not caught by `except ValueError`. The app crashes.
- Only `ValueError` is caught. Real failures include `AttributeError`, `OSError`, TensorFlow errors, and model download failures.
- No face in frame → DeepFace raises `ValueError: Face could not be detected` → caught → returns `False`. This is the one case that is handled correctly.

---

### Lines 25–40: Main loop

```python
while True:
    ret, frame = cap.read()
    if ret:
        face_match = check_face(frame)

        if face_match:
            cv2.putText(frame, "MATCH", (20, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NOT A MATCH", (20, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("j"):
        break
```

**Data flow:**
```
cap.read() → (ret, frame)
    → check_face(frame) → face_match (bool)
        → putText on frame
            → imshow("video", frame)
                → waitKey(1ms)
                    → if 'j' pressed: break
```

**Performance problem:** `check_face` calls `DeepFace.verify` synchronously. Each call takes 0.3–2 seconds on CPU. The main loop is blocked during inference — the video window freezes every frame.

**Exit key:** Pressing `j` breaks the loop. `q` and ESC do not work. This is non-standard and will confuse users.

**cv2.waitKey(1):** Waits 1 millisecond for a key press. This is correct for real-time video — but the actual frame rate is limited by DeepFace inference time, not this wait.

---

### Lines 43–44: Cleanup

```python
cap.release()
cv2.destroyAllWindows()
```

- `cap.release()` — releases the camera device so other applications can use it.
- `cv2.destroyAllWindows()` — closes the OpenCV display window.

These lines only execute if the loop exits via the `j` key. If the application crashes (exception) or the user presses Ctrl+C, cleanup is skipped and the camera may remain locked.

---

## Data flow diagram

```
Disk
 └── referenceimage2.jpg → cv2.imread → reference_img (numpy array or None)

Webcam
 └── VideoCapture(0) → cap.read() → frame (numpy array, BGR, 640x480)

frame + reference_img
 └── DeepFace.verify()
      ├── Detect faces (Haar cascade)
      ├── Align + crop (224x224)
      ├── VGG-Face embedding (4096-dim vector)
      ├── Cosine distance
      └── distance < 0.40 → verified: True/False

verified
 └── cv2.putText(frame, "MATCH" | "NOT A MATCH")
      └── cv2.imshow("video", frame)
```

---

## Summary table

| Component | Technology | Status |
|---|---|---|
| Camera capture | OpenCV VideoCapture | Works if camera available and non-headless OpenCV installed |
| Frame display | OpenCV imshow | Broken — headless build installed |
| Reference image | cv2.imread + hardcoded path | Broken — macOS path on Windows |
| Face verification | DeepFace.verify | Broken — DeepFace not installed |
| Result display | cv2.putText | Works once display is fixed |
| Exit | cv2.waitKey + ord('j') | Works but non-standard |
| Cleanup | cap.release + destroyAllWindows | Correct but only on clean exit |
