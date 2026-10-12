# Contributing to FaceVerification AI

Thank you for your interest in contributing. Please read this guide before opening a pull request.

## Ground rules

- Do not commit `.env` files, real face images, or biometric data of any kind.
- Do not weaken verification thresholds without a documented threat-model justification.
- Do not disable liveness detection in any code path — including tests.
- All new features require tests.
- Security-sensitive changes must describe the impact on the threat model in the PR description.

## Development setup

```powershell
git clone https://github.com/TechieTojin/FaceVerification.git
cd FaceVerification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend\requirements_backend.txt
cd frontend && npm install && cd ..
Copy-Item .env.example .env
# Edit .env with your own credentials
```

## Running tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/ tests/ -v
```

All 102 tests must pass before a PR is merged.

## Pull request process

1. Fork the repository and create a feature branch from `master`.
2. Make your changes with focused commits.
3. Add or update tests covering the change.
4. Run the full test suite locally.
5. Open a PR against `master` with a clear description of what changed and why.
6. PRs that touch authentication, biometric storage, or the verification pipeline require an explicit security impact section.

## What we welcome

- Bug fixes with a reproducible test case
- Documentation improvements and corrections
- Performance improvements with before/after benchmarks
- New AI model backend support (with accuracy benchmarks and threat model notes)
- Improved test coverage

## What we do not accept

- Changes that weaken security thresholds without documented justification
- New dependencies without justification (especially for the critical verification path)
- PRs that do not include tests
- Code that stores or logs raw embeddings or facial images

## Code style

- Python: follow PEP 8; type annotations on all public functions.
- TypeScript: strict mode; no `any`.
- No commented-out code in PRs.
- Comments only for non-obvious constraints or workarounds.

## Reporting security issues

Do not open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md).
