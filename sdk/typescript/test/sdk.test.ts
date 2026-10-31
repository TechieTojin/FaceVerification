import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  FaceVerificationClient, FaceVerificationError, constructWebhookEvent, verifyAssertion, verifyWebhookSignature,
  type Jwks,
} from '../src/index.ts'

// Produced by backend.services.integration_service.signature_header (Python) for the same inputs.
const VECTOR = {
  secret: 'whsec_test_vector',
  body: '{"id":"evt_1","type":"ping"}',
  header: 't=1793400000,v1=5987fa07715a002f2a8d384f5c0249cc1c12c8e8e22b7792110841854b1c1949',
  now: 1793400000,
}

test('webhook signature matches the server implementation', async () => {
  assert.equal(await verifyWebhookSignature(VECTOR.secret, VECTOR.body, VECTOR.header, { now: VECTOR.now }), true)
  assert.equal(await verifyWebhookSignature(VECTOR.secret, VECTOR.body + ' ', VECTOR.header, { now: VECTOR.now }), false)
  assert.equal(await verifyWebhookSignature('whsec_other', VECTOR.body, VECTOR.header, { now: VECTOR.now }), false)
  // Replayed outside the tolerance window.
  assert.equal(await verifyWebhookSignature(VECTOR.secret, VECTOR.body, VECTOR.header, { now: VECTOR.now + 301 }), false)
  await assert.rejects(constructWebhookEvent(VECTOR.secret, VECTOR.body, 't=1,v1=00'), FaceVerificationError)
})

function b64url(buf: ArrayBuffer | Uint8Array): string {
  return Buffer.from(buf instanceof Uint8Array ? buf : new Uint8Array(buf)).toString('base64url')
}

async function makeAssertion(claims: Record<string, unknown>) {
  const pair = await crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify'])
  const jwk = await crypto.subtle.exportKey('jwk', pair.publicKey)
  const header = b64url(new TextEncoder().encode(JSON.stringify({ alg: 'ES256', kid: 'k1', typ: 'fv-assertion+jwt' })))
  const payload = b64url(new TextEncoder().encode(JSON.stringify(claims)))
  const sig = await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, pair.privateKey,
    new TextEncoder().encode(`${header}.${payload}`))
  const jwks: Jwks = { keys: [{ kty: 'EC', crv: 'P-256', alg: 'ES256', kid: 'k1', x: jwk.x!, y: jwk.y! }] }
  return { token: `${header}.${payload}.${b64url(sig)}`, jwks }
}

test('assertions are verified for signature, audience, issuer and expiry', async () => {
  const now = 1_800_000_000
  const claims = { iss: 'https://fv.test/api/v1/integrations', aud: 'client-1', sub: 'order-42', jti: 'j1',
    iat: now, exp: now + 300, outcome: 'VERIFIED' }
  const { token, jwks } = await makeAssertion(claims)
  const got = await verifyAssertion(token, jwks, { audience: 'client-1', issuer: claims.iss, now })
  assert.equal(got.outcome, 'VERIFIED')
  await assert.rejects(verifyAssertion(token, jwks, { audience: 'client-2', now }), /another client/)
  await assert.rejects(verifyAssertion(token, jwks, { audience: 'client-1', issuer: 'https://evil', now }), /issuer/)
  await assert.rejects(verifyAssertion(token, jwks, { audience: 'client-1', now: now + 1000 }), /expired/)
  const [h, p, s] = token.split('.')
  const forged = `${h}.${b64url(new TextEncoder().encode(JSON.stringify({ ...claims, outcome: 'VERIFIED!' })))}.${s}`
  assert.ok(p)
  await assert.rejects(verifyAssertion(forged, jwks, { audience: 'client-1', now }), /signature/)
  const other = await makeAssertion(claims)
  await assert.rejects(verifyAssertion(token, other.jwks, { audience: 'client-1', now }), /signature/)
})

test('client sends credentials, idempotency keys and maps errors', async () => {
  const calls: { url: string; init: RequestInit }[] = []
  const fakeFetch = (async (url: string, init: RequestInit) => {
    calls.push({ url, init })
    if (url.endsWith('/sessions')) {
      return new Response(JSON.stringify({ id: 'v1', status: 'WAITING', verification_url: 'https://fv.test/m/x' }),
        { status: 201 })
    }
    return new Response(JSON.stringify({ detail: 'Invalid or revoked API credential', code: 'INVALID_API_KEY' }),
      { status: 401 })
  }) as unknown as typeof fetch
  const fv = new FaceVerificationClient({ baseUrl: 'https://fv.test/', apiKey: 'fvk_abc123_secretSECRET', fetch: fakeFetch })
  const v = await fv.createVerification({ subjectEmail: 'a@b.c', externalRef: 'o-1', idempotencyKey: 'k-1' })
  assert.equal(v.verification_url, 'https://fv.test/m/x')
  const headers = calls[0]!.init.headers as Record<string, string>
  assert.equal(calls[0]!.url, 'https://fv.test/api/v1/integrations/sessions')
  assert.equal(headers.Authorization, 'Bearer fvk_abc123_secretSECRET')
  assert.equal(headers['Idempotency-Key'], 'k-1')
  assert.deepEqual(JSON.parse(String(calls[0]!.init.body)), { subject_email: 'a@b.c', external_ref: 'o-1', expires_in: 600 })
  await assert.rejects(fv.getVerification('v1'), (e: unknown) =>
    e instanceof FaceVerificationError && e.status === 401 && e.code === 'INVALID_API_KEY')
})

test('refuses malformed keys and browser use', () => {
  assert.throws(() => new FaceVerificationClient({ baseUrl: 'https://x', apiKey: 'not-a-key' }))
  ;(globalThis as { window?: unknown }).window = {}
  try {
    assert.throws(() => new FaceVerificationClient({ baseUrl: 'https://x', apiKey: 'fvk_a1_b' }), /server/)
  } finally {
    delete (globalThis as { window?: unknown }).window
  }
})
