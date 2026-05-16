#!/usr/bin/env bash
# Tiny local server. Pyodide needs HTTP (not file://) and the right MIME
# types for .wasm / .whl / .mjs. Python's http.server does the right
# thing for our purposes (.whl is just a zip; .mjs/.wasm get correct
# MIME thanks to Python 3.13's defaults).
#
# Run from this directory: ./serve.sh   then open http://localhost:8080/
set -euo pipefail
cd "$(dirname "$0")/.."   # cd into web/ so /app and /wheels are siblings
python -m http.server 8080
