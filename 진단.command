#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
  echo '먼저 설치.command를 실행하세요.'
  exit 1
fi
trigger_result=0
.venv/bin/python voice_trigger.py --doctor || trigger_result=$?
if [ -t 0 ]; then read -r -p 'Enter를 누르면 종료합니다. ' unused_reply; fi
exit "$trigger_result"
