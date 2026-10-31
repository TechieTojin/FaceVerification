"""FaceVerification offline kiosk agent.

A managed kiosk runtime that verifies faces locally — with no connection to the
central server — for a small, explicitly authorised set of subjects, and later
synchronises its tamper-evident audit log.

What it is and is not:
  * Inference (landmarks, liveness challenge, passive anti-spoofing, ArcFace
    matching) runs on this machine with the same engine as the server.
  * The device key and local data key are protected by the operating system
    (Windows DPAPI, user scope) or by a passphrase. This is NOT hardware-backed
    key storage; anyone with the kiosk's OS account can use the keys.
  * Provisioned templates expire (server-enforced maximum validity). Revocation
    reaches the kiosk only when it reconnects.
"""

VERSION = "4.0.0"
