#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
  echo '먼저 설치.command를 실행하세요.'
  if [ -t 0 ]; then read -r -p 'Enter를 누르면 종료합니다. ' unused_reply; fi
  exit 1
fi
if .venv/bin/python voice_trigger.py; then
  exit 0
else
  trigger_result=$?
  echo '실행에 실패했습니다. 오류를 확인하고 필요하면 설치.command를 실행하세요.'
  if [ -t 0 ]; then read -r -p 'Enter를 누르면 종료합니다. ' unused_reply; fi
  exit "$trigger_result"
fi
