# FaceVerification v3.0 — Implementation Progress (durable log)

This file is the resume point if work is interrupted. Update it after each phase.

## Environment facts (verified 2026-10-30)
- Python 3.10.0 in `.venv`; DeepFace 0.0.102 runs on **onnxruntime** (no TensorFlow).
- Cached weights in `~/.deepface/weights`: ArcFace ONNX, MiniFASNetV2/V1SE ONNX.
- `mediapipe 1.2.0` installed into `.venv` with `--no-deps` (+ absl-py, flatbuffers, certifi, matplotlib)
  to avoid pulling `opencv-contrib-python 5.0`, which would clash with the pinned `opencv-python 4.10`.
- Landmarker model: `models/face_landmarker.task` (SHA-256 64184E22…0BC9FF, 3,758,596 bytes), gitignored.
- MediaPipe yaw sign: positive matrix yaw ⇔ nose toward image-right ⇔ subject turned to *their* left
  (unmirrored camera frame).
- Node 24, npm 11. No Docker on this machine (compose file is written but cannot be run here).

## Key design decisions
- Transport: HTTP multipart frame submission with exactly one in-flight frame per session
  (server-enforced via atomic UPDATE), server-issued one-time frame tokens + sequence numbers.
  SSE for live dashboard/alerts. Rationale: cookie+CSRF auth reuse, simple backpressure, testable.
- One capture-session pipeline for three purposes: VERIFY, ENROL, DOCUMENT.
- Server landmarks via MediaPipe FaceLandmarker (detection, yaw/pitch, blink blendshapes);
  passive anti-spoof via DeepFace Fasnet; embeddings via DeepFace ArcFace on an aligned crop
  (`detector_backend="skip"`), BGR input.
- Templates v2 = header + float32 L2-normalised vector; legacy v1 (v2.1, RGB-swapped pipeline)
  templates are kept but flagged `REENROLMENT_REQUIRED` and never used for matching.
- Policy service: versioned `policy_versions` rows; sessions snapshot the policy at creation.
- Inference: bounded ThreadPoolExecutor + non-blocking admission semaphore → 503 when saturated.
- Test-only fake inference provider, refused unless `APP_ENV=test`.

## Phase checklist

### Phase A/B/C/D backend — DONE (2026-10-30)
- config, database, models (+documents/audit/policy/alerts), Alembic 0001 baseline + 0002 v3 data migration
- core: permissions (RBAC), deps, event bus (SSE), structured logging + request IDs
- services: policy, templates (FVT2), imaging, providers (DeepFace/MediaPipe + test-only Fake), engine
  (challenge state machine), inference runtime, capture (sessions), enrolment, documents, evaluation,
  analytics, alerts, audit, health, maintenance
- routers: auth, users, enrolments, sessions, history, dashboard/analytics, evaluation, security, policy,
  documents, system (+SSE /stream)
- Backend tests: 154 passed (fake provider) + 8 real-model tests passed (RUN_REAL_MODEL_TESTS=1)

### Real-model observations (7 LFW public images — NOT a statistically meaningful evaluation)
- genuine pairs n=11 mean 0.472 max 0.701 (one genuine pair above the 0.68 default → false reject)
- impostor pairs n=10 mean 0.943 min 0.866
- MiniFASNet labelled a *scanned photograph* REAL (1.000) → passive anti-spoofing alone does not stop
  photo replays here; the active challenge is what blocks still images.
- BGR vs RGB-swapped input changes the embedding by cosine 0.034.

### Phase E frontend, QA, infrastructure — DONE (2026-10-30)
- Frontend rebuilt (design tokens, component library, camera/tracker/guidance/capture loop, 18 pages).
- Vitest 33 passed; Playwright deterministic 33 passed; real-model browser 5 passed; build/typecheck clean.
- Docker/Compose/nginx/CI written (not executable here); docs: README, SECURITY, DEPLOYMENT, V3_TEST_REPORT.
- Nothing committed or pushed. See docs/V3_TEST_REPORT.md for results and the "Not performed" list.
