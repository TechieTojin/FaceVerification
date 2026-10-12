# FaceVerification AI 2.0 — Architecture

## System Overview

```
Browser (React/Vite)
    │
    │  HTTP (proxied via Vite dev server)
    ▼
FastAPI Backend (uvicorn)
    │
    ├── SQLite Database (SQLAlchemy ORM)
    │
    └── DeepFace AI Engine
            ├── ArcFace ONNX  (~/.deepface/weights/arcface_weights.onnx)
            └── MiniFASNet    (~/.deepface/weights/2.7_80x80_MiniFASNetV2.onnx)
```

## Component Descriptions

### Backend (`backend/`)

| Component | Description |
|-----------|-------------|
| `main.py` | FastAPI app, CORS, lifespan (DB init + admin seed) |
| `config.py` | pydantic-settings: reads .env, provides typed config |
| `database.py` | SQLAlchemy engine (SQLite default, PostgreSQL via DATABASE_URL) |
| `models/` | ORM models: User, BiometricEnrolment, VerificationSession, etc. |
| `schemas/` | Pydantic request/response validation schemas |
| `routers/` | FastAPI routers (one per domain) |
| `services/` | Business logic: auth_service, biometric_service, audit_service |
| `middleware/` | Request logging |

### Frontend (`frontend/`)

| Component | Description |
|-----------|-------------|
| `App.tsx` | React Router setup, QueryClientProvider, protected routes |
| `store/authStore.ts` | Zustand auth state (persisted to localStorage) |
| `api/client.ts` | Axios instance with JWT auth header injection |
| `api/hooks/` | TanStack Query hooks for all API endpoints |
| `components/` | Layout, Sidebar, Header |
| `pages/` | Login, Dashboard, Verification, Enrolment, Users, History, Security, Analytics, Settings |

### AI Engine Integration

The biometric service (`backend/services/biometric_service.py`) wraps DeepFace:

1. **Face Detection**: `DeepFace.extract_faces()` with opencv backend
2. **Liveness**: Same call with `anti_spoofing=True` (MiniFASNet weights)
3. **Embedding**: `DeepFace.represent()` with ArcFace model → 512-dim vector
4. **Storage**: Fernet-encrypted blob in SQLite BLOB column
5. **Comparison**: Cosine distance between enrolled and live embeddings

### Verification Pipeline (per frame)

```
Frame (JPEG bytes)
    │
    ├─ Face count check
    │   ├─ 0 faces → NO_FACE (retry)
    │   └─ >1 face → MULTIPLE_FACES (reject)
    │
    ├─ Liveness check (if enabled)
    │   └─ not real → SPOOF_DETECTED (reject)
    │
    ├─ Generate embedding
    │
    ├─ Cosine distance vs. enrolled embedding
    │   ├─ distance ≤ threshold → VERIFIED
    │   └─ distance > threshold → REJECTED
    │
    └─ Record VerificationEvent in DB
```

## Data Flow

### Enrolment
1. Admin selects user
2. Records informed consent (ConsentRecord)
3. Captures webcam frame
4. Backend: detect → quality → embed → encrypt → store
5. BiometricEnrolment record active

### Verification
1. Select enrolled user → create VerificationSession (UUID)
2. Webcam loop: capture frame → POST /submit-frame
3. Backend runs pipeline → returns FrameResult
4. On VERIFIED → session closed, VerificationEvent recorded
5. On max attempts → session locked out

## Database Schema

```
users ─────────────────────── biometric_enrolments
  │                                    (embedding_encrypted BLOB)
  │
  ├── verification_sessions
  │       └── verification_events
  │
  ├── security_events
  │
  └── consent_records

system_settings (key-value policy store)
```

## Security Architecture

- Passwords: bcrypt (passlib, cost factor 12)
- Sessions: JWT (HS256, 60-minute expiry)
- Embeddings: Fernet (AES-128-CBC + HMAC-SHA256) at rest
- Fail-closed: every exception in verification pipeline → ERROR, never VERIFIED
- No raw frames stored
- No raw embeddings in API responses
- Anti-spoofing enabled by default
