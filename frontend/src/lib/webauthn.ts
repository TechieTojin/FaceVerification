/** Thin wrappers over @simplewebauthn/browser for the server's {challenge_id, options} envelopes. */
import { browserSupportsWebAuthn, startAuthentication, startRegistration } from '@simplewebauthn/browser';
import type { PublicKeyCredentialCreationOptionsJSON, PublicKeyCredentialRequestOptionsJSON } from '@simplewebauthn/browser';

export interface Envelope<T> { challenge_id: string; options: T }

export const passkeysSupported = () => browserSupportsWebAuthn();

export async function createPasskey(env: Envelope<PublicKeyCredentialCreationOptionsJSON>) {
  return startRegistration({ optionsJSON: env.options });
}

export async function getPasskey(env: Envelope<PublicKeyCredentialRequestOptionsJSON>) {
  return startAuthentication({ optionsJSON: env.options });
}

/** Map browser WebAuthn errors to readable messages (the user cancelling is not an error worth shouting about). */
export function passkeyErrorMessage(err: unknown): string {
  const name = (err as { name?: string })?.name ?? '';
  if (name === 'NotAllowedError') return 'The passkey request was cancelled or timed out.';
  if (name === 'InvalidStateError') return 'This authenticator is already registered.';
  if (name === 'SecurityError') return 'Passkeys are not allowed on this address. Use the configured site URL (not an IP address).';
  if (name === 'NotSupportedError') return 'This browser or device does not support passkeys.';
  return (err as Error)?.message || 'Passkey operation failed.';
}
