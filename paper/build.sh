#!/bin/bash
cd "$(dirname "$0")"
~/.local/bin/tectonic main.tex 2>&1 | grep -E "error|Writing|warning" | tail -10
ls -la main.pdf 2>/dev/null && echo "→ open: $(pwd)/main.pdf"
