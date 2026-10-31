"""Kiosk command line.

  python -m kiosk provision --server https://fv.example.com --code ABCD1234EFGH
  python -m kiosk serve [--port 8300] [--ui-dir frontend/dist]
  python -m kiosk sync | status | wipe

Options: --data-dir (default ./kiosk_data), --inference deepface|fake (fake = tests only),
         --protection dpapi|passphrase (passphrase reads KIOSK_PASSPHRASE).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys

from kiosk import VERSION, runtime_env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kiosk")
    ap.add_argument("command", choices=["provision", "serve", "sync", "status", "wipe"])
    ap.add_argument("--data-dir", default="kiosk_data")
    ap.add_argument("--server")
    ap.add_argument("--code")
    ap.add_argument("--protection")
    ap.add_argument("--inference", default="deepface", choices=["deepface", "fake"])
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--ui-dir", default="frontend/dist")
    ap.add_argument("--sync-interval", type=float, default=30.0)
    args = ap.parse_args(argv)

    runtime_env.prepare(args.inference)
    from kiosk import client, keystore
    from kiosk.store import Identity, KioskStore

    if args.command == "wipe":
        shutil.rmtree(args.data_dir, ignore_errors=True)
        print("Kiosk data removed.")
        return 0

    store = KioskStore(args.data_dir)
    if args.command == "provision":
        if store.has_identity():
            print("This kiosk is already provisioned. Run `wipe` first to re-provision.", file=sys.stderr)
            return 2
        if not args.server or not args.code:
            print("--server and --code are required", file=sys.stderr)
            return 2
        mode = args.protection or keystore.default_mode()
        key, data_key = store.create_identity(mode)
        keystore.unprotect(mode, keystore.protect(mode, b"self-test"))       # fail early if protection is unusable
        info = client.provision(args.server, args.code, key, VERSION, f"{platform.system()} {platform.release()}",
                                mode)
        store.save_identity(Identity(device_id=info["device_id"], server_url=args.server, protection=mode,
                                     data_key=data_key, private_key=key, name=info.get("name", ""),
                                     organization=info.get("organization", "")))
        print(json.dumps({"provisioned": True, "device_id": info["device_id"], "name": info.get("name"),
                          "organization": info.get("organization"), "key_protection": keystore.DESCRIPTIONS[mode]}))
        from kiosk.agent import KioskAgent
        print(json.dumps(KioskAgent(store).sync(True)))
        return 0

    if not store.has_identity():
        print("Kiosk is not provisioned. Run `python -m kiosk provision` first.", file=sys.stderr)
        return 2
    from kiosk.agent import KioskAgent
    agent = KioskAgent(store)
    if args.command == "sync":
        print(json.dumps(agent.sync(True), default=str))
    elif args.command == "status":
        print(json.dumps(agent.status(), default=str, indent=2))
    elif args.command == "serve":
        import uvicorn
        from kiosk.app import create_app
        uvicorn.run(create_app(agent, args.ui_dir, args.sync_interval), host="127.0.0.1", port=args.port,
                    log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
