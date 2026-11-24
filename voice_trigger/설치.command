#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
failed() {
  result=$?
  if [ "$result" -ne 0 ]; then
    echo '설치를 완료하지 못했습니다. 위 오류와 인터넷 연결을 확인한 뒤 다시 실행하세요.'
    if [ -t 0 ]; then read -r -p 'Enter를 누르면 종료합니다. ' unused_reply; fi
  fi
}
trap failed EXIT
if [ -x /opt/anaconda3/bin/python ] && /opt/anaconda3/bin/python -c 'from importlib.metadata import version; version("mlx-whisper")' 2>/dev/null; then
  trigger_python=/opt/anaconda3/bin/python
else
  trigger_python="$(command -v python3)"
fi
"$trigger_python" -c 'import platform,sys; assert platform.system()=="Darwin" and platform.machine()=="arm64", "Apple Silicon Mac이 필요합니다"; assert sys.version_info >= (3, 10), "Python 3.10 이상이 필요합니다"'
if [ ! -x .venv/bin/python ]; then
  "$trigger_python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip --disable-pip-version-check install --retries 1 --timeout 15 -r requirements.txt
# 공유하는 Anaconda의 무관한 앱까지 검사하는 pip check 대신 실제 런타임을 아래에서 검사한다.
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python voice_trigger.py --prepare
printf '\n설치와 모델 준비가 끝났습니다. 실행.command를 열어 주세요.\n'
