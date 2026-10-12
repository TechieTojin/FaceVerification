# Security Assessment — FaceVerification v1.3

## Threat model

This system is a **local desktop prototype** that authenticates a single registered person via webcam. It is **not** a production-grade authentication system. This document identifies what is addressed, what is partially addressed, and what remains unmitigated.

---

## Addressed threats

### 1. Fail-closed state machine
- All 10 states are explicitly defined
- Only `VERIFIED` grants authentication
- `is_authenticated()` checks both state AND freshness
- Any exception in the inference pipeline → ERROR (never VERIFIED)
- VERIFIED expires after `RESULT_EXPIRY_MS` (default 1500ms) — re-authentication required continuously

### 2. Liveness / anti-spoofing (passive PAD)
- DeepFace 0.0.102 `extract_faces(anti_spoofing=True)` uses a trained anti-spoofing model
- Returns `is_real` (bool) and `antispoof_score` (float)
- `is_real=None` or `is_real=False` → SPOOF_DETECTED (fail-closed)
- The anti-spoofing model used by DeepFace is based on a shallow CNN trained to distinguish live vs. printed/screen faces
- Configurable: `ENABLE_LIVENESS=false` disables it (disabled mode intended only for development/testing)

### 3. Multiple-face protection
- If more than one face is detected → MULTIPLE_FACES → not authenticated
- Prevents one person holding a photo of the authorised person in front of the camera alongside themselves

### 4. No face protection
- Zero detected faces → NO_FACE → not authenticated
- A blank camera or covered lens cannot be authenticated

### 5. Stale result expiry
- VERIFIED state has a 1500ms lifetime
- Old VERIFIED results from a previous session cannot be reused
- Background inference results cannot authenticate a different session (single-process, single-session design)

### 6. Reference image protection
- Path is loaded from env var / .env file, never hardcoded
- .gitignore excludes all .jpg/.png files so reference images cannot be committed

### 7. Minimal biometric logging
- No webcam frames are saved to disk
- No facial embeddings are persisted
- Only non-sensitive diagnostic information is logged (state transitions, inference time)

---

## Partially addressed threats

### Printed photograph attack
**Status: Partially mitigated**

The DeepFace anti-spoofing model can detect many printed photo attacks under normal lighting. However:
- No independent evaluation of this model's FAR/FRR on printed photos has been performed in this project
- Performance varies with lighting, print quality, and camera type
- A high-quality print in good lighting may defeat the model

### Screen/phone replay attack
**Status: Partially mitigated**

The same anti-spoofing model is the only defence. Screen replays are harder to detect than prints due to similar luminance patterns to live faces.

---

## Unmitigated threats

### Virtual camera / digital injection
**Status: NOT mitigated**

An attacker with local machine access can replace the camera feed with a pre-recorded video using OBS virtual camera or similar tools. No camera authenticity check is implemented. This is a software-level architectural problem.

**Mitigation path:** Hardware attestation, signed camera feeds — requires hardware support beyond this prototype.

### High-quality 3D mask attack
**Status: NOT mitigated**

A well-made silicone mask may defeat RGB-only liveness detection. Depth sensors (Intel RealSense, Azure Kinect) would be required.

### Adversarial examples
**Status: NOT mitigated**

Carefully crafted images can defeat neural network face recognisers. No adversarial robustness testing has been performed.

### Side-channel / memory attacks
**Status: NOT mitigated**

Reference image is held in process memory. A privileged attacker with access to the process could extract it.

### Timing side-channel
**Status: NOT mitigated**

Inference time varies between VERIFIED and REJECTED outcomes. This is unlikely to be exploitable in the current threat model.

---

## Security parameters

| Parameter | Default | Effect |
|---|---|---|
| `ENABLE_LIVENESS` | `true` | Disabling removes anti-spoofing — do not disable in any non-development use |
| `RESULT_EXPIRY_MS` | `1500` | Shorter = more secure, more re-authentication burden |
| `INFERENCE_INTERVAL_MS` | `500` | Rate-limits inference; longer interval = slower response to spoofing |

---

## Residual risks summary

| Risk | Severity | Status |
|---|---|---|
| High-quality photo spoofing | High | Partially mitigated |
| Screen/replay attack | High | Partially mitigated |
| Virtual camera injection | Critical | Not mitigated |
| 3D mask attack | High | Not mitigated |
| No attempt limiting | Medium | Not implemented (v2.0) |
| No encryption of reference images | Medium | Not implemented (v2.0) |

---

## Conclusion

FaceVerification v1.3 is suitable for **controlled demonstration and development testing**. It must not be used as the sole authentication mechanism for any system protecting sensitive resources without:

1. Independent evaluation of the anti-spoofing model against the actual threat population
2. Additional controls against virtual camera injection
3. Risk assessment appropriate to the deployment context
4. Legal review for biometric data regulations applicable to the deployment jurisdiction
