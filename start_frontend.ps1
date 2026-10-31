# FaceVerification 3.0 — web UI dev server (run from the repository root: .\start_frontend.ps1)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root "frontend"
Set-Location $frontend
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
}
Write-Host "Starting FaceVerification 3.0 UI on http://localhost:5173 (API must run on port 8000)" -ForegroundColor Cyan
npm run dev
