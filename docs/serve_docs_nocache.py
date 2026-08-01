#!/usr/bin/env python3
"""Serve the built Sphinx docs with cache-disabled headers."""
from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "build" / "html"
PORT = 8000


class NoCacheHandler(SimpleHTTPRequestHandler):
    def _redirect_legacy_paths(self):
        if self.path.split("?", 1)[0] == "/tutorials/he_nissel_alignment.html":
            self.send_response(301)
            self.send_header("Location", "/tutorials/st_histology_image_processing.html")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Clear-Site-Data", '"cache"')
            self.end_headers()
            return True
        return False

    def do_GET(self):
        if self._redirect_legacy_paths():
            return
        super().do_GET()

    def do_HEAD(self):
        if self._redirect_legacy_paths():
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Clear-Site-Data", '"cache"')
        super().end_headers()


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError("Build the documentation first: make -C docs html")
    os.chdir(ROOT)
    ThreadingHTTPServer(("", PORT), NoCacheHandler).serve_forever()


if __name__ == "__main__":
    main()
