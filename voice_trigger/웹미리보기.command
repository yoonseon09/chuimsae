#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec ./웹실행.command --preview "$@"
