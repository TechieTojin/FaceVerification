# FaceVerification 3.0 — backend (run from the repository root: .\start_backend.ps1)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found. Run: python -m venv .venv; .venv\Scripts\pip install -r backend\requirements_backend.txt; .venv\Scripts\pip install --no-deps -r backend\requirements_mediapipe.txt"
    exit 1
}
Set-Location $root
if (-not (Test-Path (Join-Path $root "models\face_landmarker.task"))) {
    Write-Host "Fetching model files..." -ForegroundColor Yellow
    & $python scripts\fetch_models.py
}
Write-Host "Starting FaceVerification 3.0 API on http://localhost:8000 (docs: /docs, readiness: /api/v1/health/ready)" -ForegroundColor Cyan
Write-Host "Migrations run automatically; SQLite databases are backed up to .\backups first." -ForegroundColor DarkGray
& $python -m uvicorn backend.main:app --port 8000 --host 127.0.0.1
