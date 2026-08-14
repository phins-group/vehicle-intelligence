"""Minimal health endpoint for the versioned training container image."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _HealthHandler(BaseHTTPRequestHandler):
    server_version = "PHINSTrainingImage/1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
        if self.path not in {"/", "/health"}:
            self.send_error(404)
            return
        body = json.dumps(
            {
                "status": "ok",
                "image": "phins-picodet-trainer",
                "paddleDetectionRevision": os.environ.get(
                    "PHINS_PADDLEDETECTION_REVISION"
                ),
            },
            sort_keys=True,
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 7860), _HealthHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
