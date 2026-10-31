# FaceVerification TypeScript SDK

Server-side client for the FaceVerification integration API (hosted face verification, signed results,
signed webhooks). Zero runtime dependencies; uses the WebCrypto and `fetch` built into Node ≥ 20.

> Keep your API key (`fvk_…`) and webhook secret (`whsec_…`) on your server. The SDK refuses to run in a
> browser unless you pass `dangerouslyAllowBrowser: true`.

## Flow

1. An administrator creates an integration client in FaceVerification (**Developers → API clients**).
   The API key is shown once. Optionally add a webhook endpoint; its signing secret is shown once.
2. Your server calls `createVerification` and sends `verification_url` to your user.
3. The user opens the link on their own device and completes the live face check (camera, server-side
   liveness challenge, matching against their enrolled template).
4. You receive `verification.completed` (and `verification.reviewed` if a reviewer decided the case) on your
   webhook, or poll `getVerification`. Both carry an ES256-signed **assertion**.
5. Verify the assertion (`verifyAssertion` locally against the JWKS, or `introspectAssertion` on the server,
   which succeeds only once) and act on `outcome`.

```ts
import { FaceVerificationClient, constructWebhookEvent } from '@faceverification/sdk'

const fv = new FaceVerificationClient({ baseUrl: 'https://fv.example.com', apiKey: process.env.FV_API_KEY! })

const v = await fv.createVerification({ subjectEmail: 'ana@example.com', externalRef: 'order-42',
                                        idempotencyKey: 'order-42-attempt-1' })
console.log(v.verification_url)

// webhook handler (use the *raw* body)
const event = await constructWebhookEvent(process.env.FV_WEBHOOK_SECRET!, rawBody, req.headers['fv-signature'])
const claims = await fv.verifyAssertion(event.data.assertion!, { audience: process.env.FV_CLIENT_ID! })
if (claims.outcome === 'VERIFIED' && claims.review_status !== 'REJECTED') { /* continue */ }
```

## Assertion claims

| Claim | Meaning |
|---|---|
| `iss` | `<base url>/api/v1/integrations` |
| `aud` | your integration client id |
| `sub` | your `externalRef`, or the subject's user id |
| `jti` | unique assertion id — store it to reject replays |
| `iat`, `exp` | issue time and expiry (default 5 minutes) |
| `outcome` | `VERIFIED`, `NOT_MATCHED`, `LIVENESS_FAILED`, `ATTEMPTS_EXHAUSTED`, `SESSION_EXPIRED`, `ERROR` |
| `review_status` | `PENDING`, `APPROVED`, `REJECTED`, `RETRY_REQUESTED`, `EXPIRED` or `null` — an administrative disposition; it never changes `outcome` |
| `liveness` | `passed` / `failed` / `not_completed` |
| `policy_version` | the organisation policy version the session ran under |

No biometric data (images, templates, embeddings, scores) is ever returned by the API.

## Webhook signatures

`FV-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256(secret, "<t>.<raw body>")>`. Signatures older than
5 minutes are rejected by `verifyWebhookSignature`. Deliveries are retried with back-off (1, 5, 30, 120,
360 minutes; 6 attempts). Each delivery has a unique `FV-Event-Id`; handle duplicates idempotently.

## Errors

All failures throw `FaceVerificationError` with `status` and `code` (`INVALID_API_KEY`, `INSUFFICIENT_SCOPE`,
`RATE_LIMITED`, `SUBJECT_NOT_FOUND`, `NOT_ENROLLED`, `IDEMPOTENCY_KEY_REUSED`, `TIMEOUT`, …).

## Development

```bash
npm install          # TypeScript only (dev dependency)
npm test             # node:test, runs the TypeScript sources directly
npm run build        # emits dist/
node examples/server.ts   # example relying party (see the header of the file for env vars)
```
