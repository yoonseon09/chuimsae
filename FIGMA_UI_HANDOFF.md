# 추임새 Figma UI 작업 완료

2026-11-24: 기존 웹 UI에 아래 세 화면을 구현했다. Figma 원본 파일은 수정하지 않았다.

- 대기: https://www.figma.com/design/LFHc11AAEuAWVdHP7wyauK/SCG?node-id=25-2
- 음성 입력: https://www.figma.com/design/LFHc11AAEuAWVdHP7wyauK/SCG?node-id=31-47
- 인식 완료: https://www.figma.com/design/LFHc11AAEuAWVdHP7wyauK/SCG?node-id=31-54

## 구현

- 세 노드의 design context와 스크린샷을 확인하고 기존 HTML/CSS/JavaScript 구조에 적용했다.
- Figma 원본 PNG 세 개를 `web/assets/`에 저장했다. 세 화면에서 제공된 각 이미지가 동일한 SHA-256임을 확인해 자산을 공유한다. 임시 Figma URL을 런타임에 사용하지 않는다.
- 일월오봉도, 회전한 소나무 배경, 51% 검정 오버레이, 전경 나무, MaruBuri 문구를 반영했다. 1180×820을 기준으로 배치하고 휴대폰 화면에 대응했다.
- 누르는 동안 좌우 ±4도 흔들림과 옅은 흰색 그림자를 적용했다. 권한 대기 중에도 누름 피드백을 보이며 손을 떼거나 pointercancel, lostpointercapture, focus/visibility 상실, 서버 오류가 발생하면 끝난다. 키보드 Space/Enter와 Pointer Events 기반 마우스·터치를 지원한다. 모션 감소 설정에서는 흔들림 대신 정적인 기울임과 그림자로 누름을 표시한다.
- 상단 완료 문구는 `recognized.text`를 사용한다. 인식 메시지가 없으면 `playback.command`를 사용한다. 재생 명령이 원래 인식 문장을 덮어쓰지 않으며 서버가 ready로 돌아와도 다음 입력까지 유지한다. 고정 예시 문구를 인식 결과로 사용하지 않는다.
- 기존 마이크 권한 대기 중 해제, 물리 마이크 정지, PCM 전송·flush, 취소, WebSocket 인증·시간 동기화 및 Mac 재생 처리는 유지했다.
- 우측 하단 연결 상태 버튼을 사용자 요청으로 완전히 제거했다. hover나 키보드 focus로도 나타나지 않는다. 하단 인식 결과·미인식 안내 패널도 사용자 요청으로 제거했다. 결과 영역은 화면에 보이지 않는 접근성 알림으로만 유지한다. 연결 오류 시 나무를 누르면 도움말을 연다.
- `web_voice.py`에 이미지 세 파일만 명시적으로 허용했다. 디렉터리 전체 정적 서빙은 추가하지 않았다.
- `web_make_preview.py`는 글꼴과 이미지 세 개를 data URL로 포함한다. 생성된 `화면미리보기.html`은 독립적으로 열리며 마이크나 재생을 사용하지 않는다. 누름 모습은 마우스와 키보드로 검토할 수 있다.

## 검증

- `node --test tests/web_client.test.cjs`: 11개 통과. 기존 6개에 누름 상태·권한 대기 중 해제·터치 취소·포커스/가시성 상실·키보드 입력·동적 인식 결과 유지 검사를 보강했다.
- `.venv/bin/python -m unittest discover -s tests -q`: 93개 통과.
- 실제 브라우저: 1180×820의 대기·음성 입력·인식 완료 배치, 390×844의 휴대폰 배치와 가로 넘침 없음, 이미지 세 개의 로드/원본 크기/렌더링 위치, 내장 이미지가 포함된 단독 HTML, 키보드 누름/해제를 확인했다.
- 완료 화면 전경 이미지 실제 경계: x=364.994, y=423.459, w=469.074, h=399.278. Figma 목표 x=365, y=423.45, w=469.075, h=399.286에 일치한다.
- 가상 오디오와 임시 로컬 WebSocket을 사용한 브라우저 검증에서 인식된 `잘한다!`가 실제 앱 JavaScript를 통해 상단에 반영되고 ready 상태 이후에도 유지됨을 확인했다. 이 테스트 장치는 `work/`에만 있으며 제품에는 포함하지 않았다.
- 이미지 세 경로 HTTP 200, `/assets/`, `/fonts/`, `../commands.json` HTTP 404 확인.
- 실제 마이크·음성 모델·Mac 스피커의 실기기 종단 간 검증은 수행하지 않았다.

## 변경 파일과 복구

- `web/index.html`, `web/style.css`, `web/app.js`
- `web/assets/royal-landscape.png`, `web/assets/pine-branches.png`, `web/assets/responding-pine.png`
- `web_voice.py`, `web_backend.py`의 나무 안내 문구, `web_make_preview.py`, `화면미리보기.html`
- `tests/web_client.test.cjs`, 이 문서
- 이번 작업 전 파일: `backups/figma-ui-20261124-resume/`
- 앞선 글꼴 작업 전 파일: `backups/figma-ui-20261124-170825/`

실행 방법은 기존 `웹실행.command`를 사용한다. 실행 중인 서버가 있으면 종료 후 다시 실행해야 Python의 이미지 경로 추가가 반영된다. `웹미리보기.command` 또는 `화면미리보기.html`은 녹음 없는 디자인 검토용이다.
