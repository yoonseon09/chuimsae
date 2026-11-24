#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
  echo '먼저 설치.command를 실행하세요.'
  if [ -t 0 ]; then read -r -p 'Enter를 누르면 종료합니다. ' unused_reply; fi
  exit 1
fi
if ! .venv/bin/python -c 'import aiohttp' >/dev/null 2>&1; then
  echo '웹 연결 라이브러리를 준비합니다.'
  .venv/bin/python -m pip --disable-pip-version-check install --retries 1 --timeout 15 'aiohttp>=3.11.10,<4'
fi
echo '기존 실행.command는 종료하고, 이 창은 켜 둔 채 사용하세요.'
if .venv/bin/python web_voice.py "$@"; then
  exit 0
else
  web_result=$?
  echo '웹 실행을 완료하지 못했습니다. 위 오류를 확인해주세요.'
  if [ -t 0 ]; then read -r -p 'Enter를 누르면 종료합니다. ' unused_reply; fi
  exit "$web_result"
fi
