# Test Plan — FaceVerification

## Test Environment

- OS: Windows 11
- Python: 3.10.0
- Framework: pytest
- Camera: Required for integration tests (can mock for unit tests)

---

## Unit Tests

### UT-01: check_face returns False when reference_img is None

```python
def test_check_face_none_reference(monkeypatch):
    monkeypatch.setattr("face_recognition.reference_img", None)
    result = check_face(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is False
```

**Expected:** Returns False without crashing  
**Current status:** FAILS — raises AttributeError (BUG-03)

---

### UT-02: check_face returns False when no face in frame

```python
def test_check_face_no_face(monkeypatch):
    def mock_verify(*args, **kwargs):
        raise ValueError("Face could not be detected")
    monkeypatch.setattr(DeepFace, "verify", mock_verify)
    result = check_face(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is False
```

**Expected:** Returns False  
**Current status:** PASSES (ValueError is caught)

---

### UT-03: check_face returns True when faces match

```python
def test_check_face_match(monkeypatch):
    monkeypatch.setattr(DeepFace, "verify", lambda *a, **k: {"verified": True})
    result = check_face(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is True
```

**Expected:** Returns True  
**Current status:** FAILS — AttributeError on reference_img.copy()

---

### UT-04: Reference image validation

```python
def test_missing_reference_image(tmp_path):
    result = cv2.imread(str(tmp_path / "nonexistent.jpg"))
    assert result is None
```

**Expected:** None returned (not an exception)  
**Current status:** PASSES (cv2.imread behaviour is correct; the bug is not checking for None)

---

## Integration Tests (require camera and reference image)

### IT-01: Application starts without error

**Preconditions:**  
- Virtual environment active  
- Dependencies installed  
- Reference image exists and REFERENCE_IMAGE_PATH set  
- Camera connected  

**Steps:**  
1. Run `python face-recognition.py`

**Expected:** Window opens showing webcam feed  
**Pass criteria:** No exception on startup; video window visible within 10 seconds

---

### IT-02: MATCH — authorised person's face

**Steps:**  
1. Start application  
2. Present the face matching the reference image to the camera

**Expected:** Green "MATCH" text on screen  
**Pass criteria:** MATCH displayed within 5 seconds of face being clearly visible

---

### IT-03: NOT A MATCH — different person

**Steps:**  
1. Start application  
2. Present a different person's face

**Expected:** Red "NOT A MATCH" text  
**Pass criteria:** NOT A MATCH displayed; MATCH never displayed for the wrong face

---

### IT-04: No face visible

**Steps:**  
1. Start application  
2. Cover the camera

**Expected:** "NOT A MATCH" or no result text  
**Pass criteria:** Application does not crash; continues running

---

### IT-05: Reference image missing

**Steps:**  
1. Set REFERENCE_IMAGE_PATH to a non-existent file  
2. Run application

**Expected:** Clear error message; application exits cleanly  
**Pass criteria:** Exit with non-zero code and descriptive message (after BUG-03 fix)

---

### IT-06: Camera unavailable

**Steps:**  
1. Disconnect or disable camera  
2. Run application

**Expected:** Clear error message; application exits cleanly  
**Pass criteria:** Exit with non-zero code and descriptive message (after BUG-02 fix)

---

### IT-07: Clean exit

**Steps:**  
1. Start application  
2. Press `q`

**Expected:** Application closes; camera released  
**Pass criteria:** No lingering processes; camera accessible to other applications after exit

---

### IT-08: Photo spoofing (security test)

**Steps:**  
1. Print a photo of the reference person  
2. Hold it in front of the camera

**Expected (with liveness):** NOT A MATCH  
**Current expected (without liveness):** MATCH — this is a known critical weakness  
**Pass criteria (Version 1.3+):** Printed photo does not produce MATCH

---

## Manual Testing Checklist

| Scenario | Expected | Pass/Fail | Notes |
|---|---|---|---|
| Cold start, all deps installed | App opens | | |
| Authorised person, good lighting | MATCH | | |
| Authorised person, dim lighting | MATCH or graceful fail | | |
| Authorised person, face obstructed | NOT A MATCH | | |
| Stranger's face | NOT A MATCH | | |
| No face (camera covered) | No crash, NOT A MATCH | | |
| Multiple faces | No crash, defined behaviour | | |
| Camera unplugged mid-session | Graceful error, no crash | | |
| Press q | Clean exit | | |
| Press ESC | Clean exit (after fix) | | |

---

## Acceptance Criteria for Version 1.1

- [ ] All unit tests pass with pytest
- [ ] IT-05 passes (missing reference image handled)
- [ ] IT-06 passes (no camera handled)
- [ ] IT-07 passes (clean exit)
- [ ] IT-01 passes (startup)
- [ ] No unhandled exceptions in any scenario above
