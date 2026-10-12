# FaceVerification AI 2.0 — Backend Startup Script
# Run from repository root: .\start_backend.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venv)) {
    Write-Error "Virtual environment not found at $venv. Run: python -m venv .venv && .venv\Scripts\pip install -r backend\requirements_backend.txt"
    exit 1
}

Write-Host "Starting FaceVerification AI 2.0 backend..." -ForegroundColor Cyan
Write-Host "  API docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  Health:   http://localhost:8000/api/v1/health" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Yellow

Set-Location $root
& $venv -m uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0
