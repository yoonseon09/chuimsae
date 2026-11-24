# 추임새 Figma UI 작업 상태

사용자 요청: 아래 세 Figma 화면으로 기존 웹 UI를 바꾼다. 기존 마이크·음성 인식·Mac 스피커 재생 기능은 유지한다.

- https://www.figma.com/design/LFHc11AAEuAWVdHP7wyauK/SCG?node-id=25-2&m=dev
- https://www.figma.com/design/LFHc11AAEuAWVdHP7wyauK/SCG?node-id=31-47&m=dev
- https://www.figma.com/design/LFHc11AAEuAWVdHP7wyauK/SCG?node-id=31-54&m=dev

나무를 누르는 동안 좌우 3~5도(기준 4도)로 흔들고 그림자를 옅은 흰색으로 바꿔 누름을 표시한다. 손을 떼거나 입력이 취소되면 누름 효과를 끝낸다. 키보드와 터치 입력 모두 지원한다. 세 번째 화면 상단에는 예시 고정 문구가 아닌 실제 인식된 추임새를 표시한다.

## 완료

- 사용자가 지정한 `/Users/yoonseon/Downloads/마루 부리/MaruBuriTTF`의 Regular·SemiBold·Bold를 손실 없는 WOFF2로 변환해 `web/fonts`에 저장했다.
- `web/style.css`에 MaruBuri 글꼴을 적용했다.
- `web_voice.py`에 글꼴 정적 경로를 추가했다.
- `web_make_preview.py`가 글꼴을 data URL로 포함하도록 변경했다. 단독 HTML 미리보기에서도 외부 연결 없이 같은 글꼴을 사용한다.
- 변경 전 파일은 `backups/figma-ui-20261124-170825`에 보관했다.
- 검증: Python 전체 93개 검사와 JavaScript 6개 검사 통과. 글꼴 세 파일의 명시적 경로 응답과 한글 글리프, 단독 미리보기의 글꼴 포함을 확인했다. 실제 브라우저 화면과 마이크·스피커 실기기 검증은 수행하지 않았다.

## 미완료와 차단 사유

사용자가 Figma 설치와 계정 연결을 완료했다고 확인했다. 설치 상태도 도구로 확인했다. 그러나 현재 대화의 도구 목록에는 Figma MCP 도구가 전혀 노출되지 않는다. 웹 도구도 Figma 링크를 열지 못했다. 실제 디자인을 아직 보지 못했으므로 화면 교체·나무 자산·흔들림·상단 인식 결과 배치는 구현하지 않았다.

새 대화에서 Figma 도구를 확인하고 위 세 노드의 get_design_context와 스크린샷을 먼저 읽는다. 사용자가 이미 계정 연결을 완료했으므로 재설치를 요청하지 않는다. 다음 스킬을 따른다: `/Users/yoonseon/.codex/plugins/cache/openai-curated-remote/figma/20.0.0/skills/figma-design-to-code/SKILL.md`.

## 구현 시 참고

- 현재 UI: `web/index.html`, `web/style.css`, `web/app.js`.
- `web/app.js`의 `orb` 버튼이 누르고 있는 동안 녹음하는 입력이다. 녹음/종료/취소와 권한 대기 중 해제 동작을 보존한다.
- WebSocket `recognized` 메시지의 `msg.text`는 인식 텍스트, `playback` 메시지의 `msg.command`는 재생 추임새다. 실제 메시지에 맞춰 상단 결과를 갱신한다.
- `web_voice.py`는 파일별 허용 경로만 제공한다. 글꼴도 세 파일만 명시적으로 제공한다. Figma에서 내려받는 자산의 로컬 경로도 필요한 범위만 제공한다. 기존 검사에서 디렉터리 전체 정적 서빙을 금지하므로 `add_static`을 추가하지 않는다.
- `화면미리보기.html`은 생성물이다. 원본을 수정한 뒤 `.venv/bin/python web_make_preview.py`로 갱신한다. 기존 미리보기 스크립트의 원 관련 문구와 상태도 최종 디자인에 맞춘다.
- 검사: `node --test tests/web_client.test.cjs`, `.venv/bin/python -m unittest discover -s tests -q`.
- 이 프로젝트는 휴대폰 마이크와 로컬 Mac의 음성 처리/스피커를 연결하는 로컬 앱이다. 외부 Sites 배포를 요청받지 않았다.
