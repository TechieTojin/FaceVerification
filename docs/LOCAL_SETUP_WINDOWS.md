# FaceVerification AI 2.0 — Local Setup (Windows 11)

## Prerequisites

- Python 3.10+ (https://python.org)
- Node.js 18+ (https://nodejs.org)
- Git

## Quick Start

### 1. Clone or navigate to repository
```powershell
cd "C:\Users\tojin.s\Music\Projects\FaceVerification"
```

### 2. Create Python virtual environment (if not exists)
```powershell
python -m venv .venv
```

### 3. Install Python dependencies
```powershell
.venv\Scripts\pip install -r backend\requirements_backend.txt
# Also install existing v1.3 deps if not already:
.venv\Scripts\pip install deepface==0.0.102 opencv-python==4.10.0.84 numpy==2.2.6
```

### 4. Install Node.js dependencies
```powershell
cd frontend
npm install
cd ..
```

### 5. Configure environment (optional)
```powershell
# Copy .env.example to .env and customize if needed
# DATABASE_URL defaults to SQLite: ./faceverification.db
# The backend will create the DB and admin user on first run
```

### 6. Start the backend
```powershell
.\start_backend.ps1
# OR:
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

### 7. Start the frontend (new terminal)
```powershell
.\start_frontend.ps1
# OR:
cd frontend && npm run dev
```

### 8. Open the app
- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs
- Login with the email and password you configured in `INITIAL_ADMIN_EMAIL` / `INITIAL_ADMIN_PASSWORD` in your `.env` file.

## Running Tests

### v1.3 CLI tests (existing)
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

### v2.0 backend tests
```powershell
.venv\Scripts\python.exe -m pytest backend/tests/ -v
```

## AI Model Weights

The following weights are pre-downloaded to `~/.deepface/weights/`:
- `arcface_weights.onnx` — ArcFace identity embedding model
- `2.7_80x80_MiniFASNetV2.onnx` — Anti-spoofing liveness model

If weights are missing, DeepFace will download them automatically on first use.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | `sqlite:///./faceverification.db` | Database connection string |
| SECRET_KEY | (random) | JWT signing key — set in production! |
| EMBEDDING_ENCRYPTION_KEY | (random) | Fernet key for embedding encryption |
| ENABLE_LIVENESS | `true` | Enable anti-spoofing check |
| ACCESS_TOKEN_EXPIRE_MINUTES | `60` | JWT token lifetime |
| CORS_ORIGINS | `["http://localhost:5173"]` | Allowed frontend origins |

## Production Notes

1. Set `SECRET_KEY` to a stable random value (not auto-generated)
2. Set `EMBEDDING_ENCRYPTION_KEY` to a stable Fernet key
3. Change admin password immediately after first login
4. Use PostgreSQL via `DATABASE_URL=postgresql://user:pass@host/dbname`
5. Run behind HTTPS (nginx/caddy)
6. Restrict CORS origins
