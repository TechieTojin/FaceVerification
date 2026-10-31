/**
 * Example relying-party server (Node ≥ 22, runs TypeScript directly):
 *
 *   FV_BASE_URL=http://localhost:8000 FV_API_KEY=fvk_… FV_CLIENT_ID=<client id> \
 *   FV_WEBHOOK_SECRET=whsec_… node examples/server.ts
 *
 *   POST /start {"email": "person@example.com", "orderId": "o-42"}  → returns the verification link
 *   POST /webhooks/faceverification                                   ← signed webhook from FaceVerification
 *   GET  /status/:id                                                  → polls and verifies the assertion
 *
 * The API key and webhook secret stay on this server; the browser only ever sees the link.
 */
import { createServer, type IncomingMessage } from 'node:http'

import { FaceVerificationClient, FaceVerificationError, constructWebhookEvent } from '../src/index.ts'

const env = (k: string) => {
  const v = process.env[k]
  if (!v) throw new Error(`${k} is required`)
  return v
}

const fv = new FaceVerificationClient({ baseUrl: env('FV_BASE_URL'), apiKey: env('FV_API_KEY') })
const clientId = env('FV_CLIENT_ID')
const webhookSecret = env('FV_WEBHOOK_SECRET')
const seenAssertions = new Set<string>()          // replay protection on your side (use a database in production)

function rawBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', (c) => { data += c; if (data.length > 1_000_000) req.destroy() })
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })
}

const server = createServer(async (req, res) => {
  const send = (status: number, body: unknown) => {
    res.writeHead(status, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(body))
  }
  try {
    if (req.method === 'POST' && req.url === '/start') {
      const { email, orderId } = JSON.parse(await rawBody(req))
      const v = await fv.createVerification({ subjectEmail: email, externalRef: orderId, idempotencyKey: `start-${orderId}` })
      return send(201, { id: v.id, link: v.verification_url, expires_at: v.expires_at })
    }
    if (req.method === 'POST' && req.url === '/webhooks/faceverification') {
      const body = await rawBody(req)
      const event = await constructWebhookEvent(webhookSecret, body, req.headers['fv-signature'] as string | undefined)
      if (event.data.assertion) {
        const claims = await fv.verifyAssertion(event.data.assertion, { audience: clientId })
        if (seenAssertions.has(claims.jti)) return send(200, { duplicate: true })
        seenAssertions.add(claims.jti)
        console.log(`[${event.type}] ${claims.sub}: ${claims.outcome} (review: ${claims.review_status ?? 'none'})`)
      } else {
        console.log(`[${event.type}] ${event.id}`)
      }
      return send(204, {})
    }
    const m = req.url?.match(/^\/status\/([\w-]+)$/)
    if (req.method === 'GET' && m) {
      const v = await fv.getVerification(m[1]!)
      const claims = v.assertion ? await fv.verifyAssertion(v.assertion, { audience: clientId }) : null
      return send(200, { status: v.status, outcome: v.outcome, verified: claims?.outcome === 'VERIFIED' })
    }
    send(404, { error: 'not found' })
  } catch (e) {
    const status = e instanceof FaceVerificationError && e.status >= 400 ? e.status : 400
    send(status, { error: (e as Error).message })
  }
})

server.listen(Number(process.env.PORT ?? 3005), () => console.log(`Example relying party on :${process.env.PORT ?? 3005}`))
