/**
 * FaceVerification server-side SDK (TypeScript).
 *
 * Keep API credentials on trusted servers. This SDK refuses to run in a browser
 * unless `dangerouslyAllowBrowser` is set, because an `fvk_` key in a public
 * bundle would let anyone start verifications in your organisation.
 */

export interface ClientOptions {
  /** Base URL of the FaceVerification server, e.g. https://fv.example.com */
  baseUrl: string
  /** Integration credential: fvk_<key id>_<secret> */
  apiKey: string
  fetch?: typeof fetch
  timeoutMs?: number
  dangerouslyAllowBrowser?: boolean
}

export interface CreateVerificationInput {
  subjectUserId?: string
  subjectEmail?: string
  /** Your own reference (order id, account id…). Becomes the assertion `sub`. */
  externalRef?: string
  /** Seconds until the verification link expires (60–1800). */
  expiresIn?: number
  /** Safe retries: the same key and body always return the same verification. */
  idempotencyKey?: string
}

export type VerificationStatus =
  | 'WAITING' | 'READY' | 'IN_PROGRESS' | 'COMPLETED' | 'EXPIRED' | 'CANCELLED' | 'AWAITING_CONFIRMATION'

export interface Verification {
  id: string
  status: VerificationStatus
  external_ref: string | null
  subject_id: string
  created_at: string
  expires_at: string
  completed_at: string | null
  outcome: string | null
  review_status: string | null
  attempts: number
  /** Only on creation: the link to hand to the end user. */
  verification_url?: string
  /** Only when COMPLETED: ES256-signed assertion (JWT). */
  assertion?: string | null
}

export interface AssertionClaims {
  iss: string
  aud: string
  sub: string
  jti: string
  iat: number
  exp: number
  outcome: string
  reason_code: string | null
  review_status: string | null
  verification_id: string
  external_ref: string | null
  liveness: 'passed' | 'failed' | 'not_completed'
  policy_version: number
  org: string
  channel: string
}

export interface Jwk { kty: 'EC'; crv: 'P-256'; alg: 'ES256'; kid: string; x: string; y: string; use?: string }
export interface Jwks { keys: Jwk[] }

export interface IntrospectionResult {
  active: boolean
  reason?: 'already_used' | 'invalid' | 'unknown_key'
  claims?: AssertionClaims
  jti?: string
}

export interface WebhookEvent {
  id: string
  type: 'verification.completed' | 'verification.reviewed' | 'ping'
  created_at: string
  data: Verification & { assertion?: string | null; review_status?: string | null }
}

export class FaceVerificationError extends Error {
  readonly status: number
  readonly code: string
  constructor(status: number, code: string, message: string) {
    super(message)
    this.name = 'FaceVerificationError'
    this.status = status
    this.code = code
  }
}

const enc = new TextEncoder()

function b64urlDecode(s: string): Uint8Array {
  const pad = s.length % 4 === 0 ? '' : '='.repeat(4 - (s.length % 4))
  const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/') + pad)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

function toHex(buf: ArrayBuffer): string {
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false
  let diff = 0
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i)
  return diff === 0
}

export class FaceVerificationClient {
  private readonly baseUrl: string
  private readonly apiKey: string
  private readonly fetchImpl: typeof fetch
  private readonly timeoutMs: number
  private jwksCache: { at: number; jwks: Jwks } | null = null

  constructor(options: ClientOptions) {
    if (typeof (globalThis as { window?: unknown }).window !== 'undefined' && !options.dangerouslyAllowBrowser) {
      throw new Error('FaceVerificationClient must run on a server: API keys must not ship to browsers.')
    }
    if (!/^fvk_[a-z0-9]+_[A-Za-z0-9_-]+$/.test(options.apiKey)) throw new Error('apiKey must look like fvk_<id>_<secret>')
    this.baseUrl = options.baseUrl.replace(/\/+$/, '')
    this.apiKey = options.apiKey
    this.fetchImpl = options.fetch ?? fetch
    this.timeoutMs = options.timeoutMs ?? 10_000
  }

  private async request<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {},
                           auth = true): Promise<T> {
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs)
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/api/v1/integrations${path}`, {
        method,
        signal: ctrl.signal,
        headers: {
          Accept: 'application/json',
          ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
          ...(auth ? { Authorization: `Bearer ${this.apiKey}` } : {}),
          ...headers,
        },
        body: body !== undefined ? JSON.stringify(body) : undefined,
      })
      const text = await res.text()
      const data = text ? JSON.parse(text) : {}
      if (!res.ok) {
        throw new FaceVerificationError(res.status, String(data.code ?? 'HTTP_ERROR'), String(data.detail ?? res.statusText))
      }
      return data as T
    } catch (e) {
      if (e instanceof FaceVerificationError) throw e
      if ((e as Error).name === 'AbortError') throw new FaceVerificationError(0, 'TIMEOUT', 'Request timed out')
      throw new FaceVerificationError(0, 'NETWORK_ERROR', (e as Error).message)
    } finally {
      clearTimeout(timer)
    }
  }

  /** Start a hosted verification. Send `verification_url` to your user. */
  createVerification(input: CreateVerificationInput): Promise<Verification> {
    if (!input.subjectUserId && !input.subjectEmail) throw new Error('subjectUserId or subjectEmail is required')
    return this.request<Verification>('POST', '/sessions', {
      subject_user_id: input.subjectUserId, subject_email: input.subjectEmail,
      external_ref: input.externalRef, expires_in: input.expiresIn ?? 600,
    }, input.idempotencyKey ? { 'Idempotency-Key': input.idempotencyKey } : {})
  }

  getVerification(id: string): Promise<Verification> {
    return this.request<Verification>('GET', `/sessions/${encodeURIComponent(id)}`)
  }

  /** One-time server-side validation; a second call for the same assertion returns `active: false`. */
  introspectAssertion(assertion: string): Promise<IntrospectionResult> {
    return this.request<IntrospectionResult>('POST', '/assertions/introspect', { assertion })
  }

  async fetchJwks(force = false): Promise<Jwks> {
    if (!force && this.jwksCache && Date.now() - this.jwksCache.at < 10 * 60_000) return this.jwksCache.jwks
    const jwks = await this.request<Jwks>('GET', '/.well-known/jwks.json', undefined, {}, false)
    this.jwksCache = { at: Date.now(), jwks }
    return jwks
  }

  /** Verify an assertion locally (signature, audience = your client id, issuer, expiry). */
  async verifyAssertion(token: string, opts: { audience: string; issuer?: string; clockToleranceS?: number }): Promise<AssertionClaims> {
    let jwks = await this.fetchJwks()
    const kid = JSON.parse(new TextDecoder().decode(b64urlDecode(token.split('.')[0] ?? ''))).kid
    if (!jwks.keys.some((k) => k.kid === kid)) jwks = await this.fetchJwks(true)
    return verifyAssertion(token, jwks, { audience: opts.audience,
      issuer: opts.issuer ?? `${this.baseUrl}/api/v1/integrations`, clockToleranceS: opts.clockToleranceS })
  }
}

/** Verify an ES256 assertion against a JWKS without network access. */
export async function verifyAssertion(token: string, jwks: Jwks,
                                      opts: { audience: string; issuer?: string; clockToleranceS?: number; now?: number }): Promise<AssertionClaims> {
  const parts = token.split('.')
  if (parts.length !== 3) throw new FaceVerificationError(0, 'INVALID_ASSERTION', 'Malformed assertion')
  const [h, p, s] = parts as [string, string, string]
  const header = JSON.parse(new TextDecoder().decode(b64urlDecode(h)))
  if (header.alg !== 'ES256') throw new FaceVerificationError(0, 'INVALID_ASSERTION', 'Unexpected algorithm')
  const jwk = jwks.keys.find((k) => k.kid === header.kid)
  if (!jwk) throw new FaceVerificationError(0, 'UNKNOWN_KEY', 'Assertion signed with an unknown key')
  const key = await crypto.subtle.importKey('jwk', { kty: 'EC', crv: 'P-256', x: jwk.x, y: jwk.y, ext: true },
    { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify'])
  const ok = await crypto.subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, key, b64urlDecode(s) as BufferSource,
    enc.encode(`${h}.${p}`) as BufferSource)
  if (!ok) throw new FaceVerificationError(0, 'INVALID_SIGNATURE', 'Assertion signature is not valid')
  const claims = JSON.parse(new TextDecoder().decode(b64urlDecode(p))) as AssertionClaims
  const now = opts.now ?? Math.floor(Date.now() / 1000)
  const skew = opts.clockToleranceS ?? 30
  if (claims.aud !== opts.audience) throw new FaceVerificationError(0, 'WRONG_AUDIENCE', 'Assertion is for another client')
  if (opts.issuer && claims.iss !== opts.issuer) throw new FaceVerificationError(0, 'WRONG_ISSUER', 'Unexpected issuer')
  if (claims.exp + skew < now) throw new FaceVerificationError(0, 'EXPIRED', 'Assertion has expired')
  if (claims.iat - skew > now) throw new FaceVerificationError(0, 'NOT_YET_VALID', 'Assertion issued in the future')
  return claims
}

/** Verify `FV-Signature: t=<unix>,v1=<hex HMAC-SHA256(secret, "<t>.<raw body>")>`. Use the raw request body. */
export async function verifyWebhookSignature(secret: string, rawBody: string | Uint8Array, header: string,
                                             opts: { toleranceS?: number; now?: number } = {}): Promise<boolean> {
  const fields = Object.fromEntries(header.split(',').map((kv) => {
    const i = kv.indexOf('=')
    return [kv.slice(0, i).trim(), kv.slice(i + 1).trim()]
  }))
  const t = Number(fields.t)
  if (!Number.isFinite(t) || !fields.v1) return false
  const now = opts.now ?? Math.floor(Date.now() / 1000)
  if (Math.abs(now - t) > (opts.toleranceS ?? 300)) return false
  const body = typeof rawBody === 'string' ? enc.encode(rawBody) : rawBody
  const key = await crypto.subtle.importKey('raw', enc.encode(secret) as BufferSource, { name: 'HMAC', hash: 'SHA-256' },
    false, ['sign'])
  const msg = new Uint8Array(enc.encode(`${t}.`).length + body.length)
  msg.set(enc.encode(`${t}.`))
  msg.set(body, enc.encode(`${t}.`).length)
  const mac = toHex(await crypto.subtle.sign('HMAC', key, msg as BufferSource))
  return timingSafeEqual(mac, String(fields.v1))
}

/** Verify the signature and parse the event; throws on an invalid or stale signature. */
export async function constructWebhookEvent(secret: string, rawBody: string, header: string | null | undefined,
                                            opts: { toleranceS?: number } = {}): Promise<WebhookEvent> {
  if (!header || !(await verifyWebhookSignature(secret, rawBody, header, opts))) {
    throw new FaceVerificationError(400, 'INVALID_WEBHOOK_SIGNATURE', 'Webhook signature verification failed')
  }
  return JSON.parse(rawBody) as WebhookEvent
}
