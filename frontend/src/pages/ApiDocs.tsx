import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Card, CardHeader, PageHeader } from '../components/ui/primitives';

const origin = typeof window !== 'undefined' ? window.location.origin : 'https://fv.example.com';

function Code({ children }: { children: string }) {
  return <pre className="overflow-x-auto rounded-lg bg-surface-2 p-4 text-xs leading-relaxed text-ink"><code>{children}</code></pre>;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader title={title} />
      <div className="space-y-3 p-5 text-sm leading-relaxed text-ink-2">{children}</div>
    </Card>
  );
}

export default function ApiDocs() {
  return (
    <>
      <Link to="/developers" className="mb-3 inline-flex items-center gap-1 text-sm text-muted hover:text-ink"><ArrowLeft className="h-4 w-4" />API clients</Link>
      <PageHeader title="Integration API" description="Start hosted face verifications from your server and receive signed results. Base path: /api/v1/integrations" />
      <div className="space-y-6">
        <Section title="1. Authenticate">
          <p>Send the key from <em>API clients</em> as a bearer token. Keys are scoped and rate-limited per client. Never ship a key to a browser or mobile app.</p>
          <Code>{`Authorization: Bearer fvk_<key id>_<secret>`}</Code>
        </Section>
        <Section title="2. Create a verification">
          <p>The subject must be an enrolled member of your organisation. Use an <code>Idempotency-Key</code> so retries never create duplicates.</p>
          <Code>{`curl -X POST ${origin}/api/v1/integrations/sessions \\
  -H "Authorization: Bearer $FV_API_KEY" -H "Idempotency-Key: order-42" \\
  -H "Content-Type: application/json" \\
  -d '{"subject_email": "ana@example.com", "external_ref": "order-42", "expires_in": 600}'

→ 201 {"id": "…", "status": "WAITING", "verification_url": "${origin}/m/…", "expires_at": "…"}`}</Code>
          <p>Send <code>verification_url</code> to the person. It is a one-time link; they verify with their own device's camera and the server-side liveness challenge.</p>
        </Section>
        <Section title="3. Receive the result">
          <p>Poll <code>GET /sessions/{'{id}'}</code> or subscribe a webhook. When the status is <code>COMPLETED</code> the response contains <code>assertion</code>: an ES256 JWT.</p>
          <Code>{`{ "iss": "${origin}/api/v1/integrations", "aud": "<your client id>", "sub": "order-42",
  "jti": "…", "iat": 1793400000, "exp": 1793400300, "outcome": "VERIFIED",
  "review_status": null, "liveness": "passed", "policy_version": 3, "channel": "HOSTED" }`}</Code>
          <p>Verify it against <code>GET /.well-known/jwks.json</code> (audience = your client id) and store <code>jti</code> to reject replays, or call <code>POST /assertions/introspect</code>, which returns <code>active: true</code> exactly once.</p>
          <p><code>review_status</code> is an administrative disposition by a human reviewer; it never changes <code>outcome</code>.</p>
        </Section>
        <Section title="4. Webhooks">
          <p>Events: <code>verification.completed</code>, <code>verification.reviewed</code>, <code>ping</code>. Each request carries <code>FV-Event-Id</code>, <code>FV-Event-Type</code> and:</p>
          <Code>{`FV-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256(secret, "<t>.<raw body>")>`}</Code>
          <p>Reject signatures older than 5 minutes. Failed deliveries (non-2xx or timeout) are retried after 1, 5, 30, 120 and 360 minutes; handle duplicate event ids idempotently.</p>
        </Section>
        <Section title="TypeScript SDK">
          <p>The repository includes a dependency-free server SDK in <code>sdk/typescript</code> with a runnable example relying-party server.</p>
          <Code>{`import { FaceVerificationClient, constructWebhookEvent } from '@faceverification/sdk'
const fv = new FaceVerificationClient({ baseUrl: '${origin}', apiKey: process.env.FV_API_KEY! })
const v = await fv.createVerification({ subjectEmail: 'ana@example.com', externalRef: 'order-42', idempotencyKey: 'order-42' })
const claims = await fv.verifyAssertion(assertion, { audience: process.env.FV_CLIENT_ID! })`}</Code>
        </Section>
        <Section title="Errors">
          <p>Errors are JSON <code>{'{"detail": "…", "code": "…"}'}</code>: <code>INVALID_API_KEY</code> (401), <code>INSUFFICIENT_SCOPE</code> (403), <code>SUBJECT_NOT_FOUND</code> (404),
            <code> NOT_ENROLLED</code> / <code>IDEMPOTENCY_KEY_REUSED</code> (409), <code>RATE_LIMITED</code> (429 with <code>Retry-After</code>).
            The API never returns images, templates, embeddings or scores.</p>
        </Section>
      </div>
    </>
  );
}
