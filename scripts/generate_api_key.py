"""Generate one high-entropy API key and its non-reversible config verifier."""

from __future__ import annotations

import hashlib
import secrets


def main() -> None:
    token = secrets.token_urlsafe(32)
    verifier = hashlib.sha256(token.encode("utf-8")).hexdigest()
    print("API key (store securely; shown once):")
    print(token)
    print("SHA-256 verifier (store in VIP_AUTH__PRINCIPALS):")
    print(verifier)


if __name__ == "__main__":
    main()

