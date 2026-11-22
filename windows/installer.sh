#!/usr/bin/env bash
# Lance l'installateur MI (version Python, plus lisible) : tout en séquence, idempotent.
set -euo pipefail
exec python3 "$(dirname "$0")/installer.py"
