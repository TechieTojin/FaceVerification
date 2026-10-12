# FaceVerification AI 2.0 — Frontend Startup Script
# Run from repository root: .\start_frontend.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root "frontend"

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location $frontend
    npm install
}

Write-Host "Starting FaceVerification AI 2.0 frontend..." -ForegroundColor Cyan
Write-Host "  URL: http://localhost:5173" -ForegroundColor Green
Write-Host "  Backend must be running on port 8000" -ForegroundColor Yellow
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Yellow

Set-Location $frontend
npm run dev
