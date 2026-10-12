# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| v2.0 (current) | Yes |
| v1.3 (CLI only) | Bug fixes only |
| v1.x (legacy) | No |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

To report a security issue, email the repository maintainer directly (see the GitHub profile for contact details). Include:

1. A description of the vulnerability
2. Steps to reproduce
3. Affected versions
4. Your assessment of severity and exploitability
5. Any suggested mitigations (optional)

You will receive an acknowledgement within 72 hours. If the vulnerability is confirmed, a fix will be prioritized and you will be credited (unless you prefer otherwise).

## Security model

### What this system protects

- **Identity binding**: only an enrolled face can produce a `VERIFIED` result for that user's session.
- **Liveness**: passive anti-spoofing (MiniFASNet) is applied to every frame before recognition.
- **Fail-closed**: every exception, timeout, or error path produces a non-authenticated result.
- **Embedding confidentiality**: facial embeddings are encrypted at rest (Fernet/AES-128); raw vectors are never returned by the API.
- **Credential security**: passwords are bcrypt-hashed (work factor 12); JWT tokens are signed with a configurable secret.

### Known limitations and out-of-scope attacks

The following attack classes are **not fully mitigated** by this system. Do not deploy in high-stakes environments without additional controls.

| Attack | Status | Notes |
|---|---|---|
| High-quality photo spoofing | Partially mitigated | MiniFASNet trained primarily on printed photos and screen replays; effectiveness against high-resolution prints is not independently validated |
| 3D mask attacks | Not mitigated | RGB-only passive liveness cannot detect silicone or 3D-printed masks |
| Virtual camera injection | Not mitigated | An attacker with local system access can feed synthetic video frames to the browser |
| Adversarial examples against ArcFace | Not evaluated | Not in scope for this prototype |
| Side-channel timing attacks | Not evaluated | No constant-time comparison for embeddings |
| Account lockout bypass | Partial | Session-level attempt limit enforced; no IP-level lockout |

### Not in scope (report these for awareness, not as vulnerabilities)

- Attacks requiring physical access to the server
- Attacks against the underlying OS or browser
- Social engineering of administrators
- Theoretical attacks without a working proof of concept

## Hardening checklist for production use

Before deploying in any production environment:

- [ ] Set `SECRET_KEY` to a stable, randomly generated value
- [ ] Set `EMBEDDING_ENCRYPTION_KEY` to a stable Fernet key; back it up securely
- [ ] Change the initial admin password immediately after first login
- [ ] Enable HTTPS (TLS) — never run plaintext HTTP for authenticated endpoints
- [ ] Restrict `CORS_ORIGINS` to your actual frontend domain
- [ ] Switch to PostgreSQL and set up regular backups
- [ ] Enable per-request rate limiting at the reverse proxy level
- [ ] Review and comply with applicable biometric privacy laws (GDPR, BIPA, CCPA, etc.)
- [ ] Conduct independent penetration testing before going live
- [ ] Obtain legal review of biometric data handling practices

## Privacy and biometric data

- This system processes facial images and generates biometric embeddings.
- Biometric data is subject to strict legal requirements in many jurisdictions.
- No raw video frames are stored.
- Embeddings are encrypted at rest.
- Users must provide explicit consent before enrolment (recorded in the database).
- Enrolments can be fully deleted via the API.
- **You are responsible for complying with applicable law** before collecting or processing biometric data from any individual.
