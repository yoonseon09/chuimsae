# voice_trigger — 한국어 음성인식 → 음원 재생

마이크로 말하면 한국어로 인식하고, 인식된 말에 맞는 음원 파일을 재생합니다.
Apple Silicon(M1)용 `mlx-whisper`로 Mac 안에서만 처리합니다(인터넷 불필요, 첫 실행 때 모델 다운로드만).

## 1. 설치 (처음 한 번)

```bash
cd ~/voice_trigger          # 이 폴더를 둔 위치 (경로에 한글이 없으면 좋습니다)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 실행

```bash
source .venv/bin/activate
python voice_trigger.py
```

- 처음 실행하면 모델(약 1.5GB)을 받느라 몇 분 걸립니다.
- macOS가 터미널의 **마이크 권한**을 물으면 허용하세요.
- 종료: `Ctrl + C`

### 테스트용 명령

```bash
python voice_trigger.py --list-devices          # 마이크/스피커 번호·이름 확인
python voice_trigger.py --test "안녕하세요"        # 한 문장 매칭 테스트 (소리 재생)
python voice_trigger.py --type                  # 키보드로 문장 입력하며 테스트 (모델 불필요)
python voice_trigger.py --type --no-play        # 소리 없이 매칭 결과만 보기
```

## 3. 명령 추가하기 — `commands.json`

```json
{
  "name": "박수",
  "phrases": ["박수", "박수 쳐 줘", "잘했어"],
  "sounds": ["sounds/clap.wav", "sounds/clap2.mp3"],
  "volume": 0.8,
  "cooldown": 2,
  "match": "auto"
}
```

| 항목 | 설명 |
|---|---|
| `phrases` | 이 말들 중 하나가 문장에 들어 있으면 실행. 공백·문장부호는 무시됩니다 |
| `sounds` | 재생할 파일 **또는 폴더**. 여러 개(폴더면 그 안의 모든 음원)면 **무작위로 하나** 재생하되, 직전 파일은 연속으로 안 나옴. wav, mp3, flac, ogg, aiff 가능 |
| `volume` | 볼륨 배율 (선택, 기본 1.0) |
| `cooldown` | 같은 명령을 다시 실행하기까지 기다릴 초 (선택) |
| `match` | `auto`(기본, 비슷하면 인정) · `contains`(글자 그대로 포함) · `exact`(문장 전체가 일치) |

- 경로는 commands.json 기준입니다. 같은 폴더의 파일은 `"남_얼쑤.wav"`, 하위 폴더는 `"sounds/파일명.wav"`.
- 폴더를 지정하면(`"sounds": ["남_얼쑤"]`) 그 폴더에 파일을 넣고 빼는 것만으로 목록이 바뀝니다.
- **실행 중에 저장해도 바로 반영**됩니다(재시작 불필요).
- 아무 명령에도 안 맞을 때 재생할 소리: `"fallback": { "sounds": ["sounds/what.wav"] }`

## 4. 주요 설정 — `settings`

| 항목 | 기본값 | 설명 |
|---|---|---|
| `model` | whisper-large-v3-turbo | 사용할 Whisper 모델 (아래 “모델 바꾸기” 참고) |
| `engine` | auto | `mlx` · `transformers` · `auto`(이름에 mlx 가 있으면 mlx, 아니면 transformers) |
| `input_device` / `output_device` | null(시스템 기본) | 번호 또는 이름 일부. 예: `"BlackHole"`, `"MacBook Air 스피커"` |
| `start_threshold_db` | -40 | 이보다 큰 소리를 말로 판단. 주변이 시끄러우면 -35, 인식을 못 하면 -45 |
| `silence_duration` | 0.4 | 이만큼 조용하면 말이 끝난 것으로 판단 (초). **지연시간에 그대로 더해짐** |
| `max_speech` | 4 | 한 번에 녹음하는 최대 길이 (초) |
| `match_threshold` | 75 | 0~100. 오작동이 많으면 올리고, 잘 안 걸리면 내리세요 |
| `play_mode` | interrupt | `interrupt`: 새 소리가 이전 소리를 끊음 / `skip_if_playing`: 재생 중엔 무시 |
| `ignore_while_playing` | true | 재생 중엔 마이크를 무시 (스피커 소리를 다시 인식하는 것 방지) |
| `use_phrases_as_prompt` | true | 등록한 단어를 모델에 힌트로 줘서 인식률을 높임 |
| `ignore_texts` | (목록) | Whisper가 무음에서 지어내는 문장(“시청해주셔서 감사합니다” 등) 무시 |

### 모델 바꾸기

기본값 `mlx-community/whisper-large-v3-turbo` 는 **OpenAI 원본 `openai/whisper-large-v3-turbo` 와 같은 모델**을
Apple Silicon 용 MLX 형식으로 바꾼 것입니다. 인식 결과는 같고 M1 에서 가장 빠릅니다.

Hugging Face 원본 모델을 그대로 쓰고 싶다면:

```bash
python -m pip install transformers torch     # 처음 한 번
```
```json
"model": "openai/whisper-large-v3-turbo",
"engine": "auto",
```

- `openai/whisper-small`, `openai/whisper-large-v3` 처럼 Hugging Face 에 있는 Whisper 모델은 이름만 바꾸면 됩니다.
- M1 GPU(MPS)로 돌지만 보통 MLX 보다 느립니다. 화면의 `⏱️` 지연시간을 비교해 보고 고르세요.
- 모델·엔진 변경은 **프로그램을 다시 실행**해야 반영됩니다(명령 목록은 실행 중에도 바로 반영).

| 모델 | 엔진 | 특징 |
|---|---|---|
| `mlx-community/whisper-large-v3-turbo` | mlx | 기본. 정확하고 빠름 |
| `mlx-community/whisper-small-mlx` | mlx | 더 빠름, 정확도 ↓ |
| `openai/whisper-large-v3-turbo` | transformers | 원본 그대로. 위 첫 줄과 같은 모델 |
| `openai/whisper-large-v3` | transformers | 가장 정확, 가장 느림 |

## 5. 지연시간 (목표 1~2초)

지연시간 = `silence_duration`(말 끝 판단 대기) + 음성인식 시간 (+ 재생 지연 거의 0, 음원은 미리 메모리에 올려 둠)

매번 화면에 `⏱️ 말 끝 → 재생 시작: 1.05초` 처럼 실제 지연이 표시됩니다.
- 2초를 넘으면: `"model": "mlx-community/whisper-small-mlx"` 로 바꾸기 (빠르지만 정확도 ↓). transformers 엔진을 쓰는 중이라면 먼저 mlx 로 돌아오기
- 더 줄이려면: `silence_duration` 을 0.3 으로 (너무 줄이면 말 중간에 잘림)

## 6. 다른 앱으로 소리 보내기 (디스코드, OBS, RVC 등)

`brew install blackhole-2ch` 설치 후 `"output_device": "BlackHole"`로 지정하고,
받는 앱의 마이크 입력을 BlackHole 2ch로 고르면 됩니다.

## 문제 해결

- **인식 결과가 안 뜸** → 화면에 `… 인식 중`이 안 나오면 `start_threshold_db`를 -45~-50으로 낮추기
- **말 중간에 잘림** → `silence_duration`을 0.6으로 늘리기
- **아무 소리에나 “얼쑤”가 걸림** → `use_phrases_as_prompt`를 false로 (등록 단어 힌트가 소음을 그 단어로 착각하게 만들 수 있음)
- **재생 중에 외친 추임새가 무시됨** → 이어폰·BlackHole로 출력한다면 `ignore_while_playing`을 false로
- **엉뚱한 명령 실행** → `match_threshold`를 85로 올리거나 해당 명령에 `"match": "contains"`
- **mp3 재생 오류** → `pip install -U soundfile` 또는 wav로 변환
