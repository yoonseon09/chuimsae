# 추임새 인식 라이브러리·API 조사

조사일: 2026-11-13. 공식 문서의 지원 기능을 확인한 결과이며, 실제 마이크·소음 환경의 성능 측정 결과는 아니다. 이 문서는 기존 프로그램과 설정을 변경하지 않는다.

후속 구현 결정: 사용자가 설치·구현을 요청한 뒤, API 계정 없이 시작할 수 있고 기존 Mac에 라이브러리와 모델이 있는 MLX Whisper를 첫 기본 엔진으로 선택했다. Silero VAD와 결합한 코드를 작성했다. 아래의 CLOVA 권고는 외부 API 비교 후보의 우선순위다. 실제 설치·검증 상태는 [INSTALL_STATUS.md](INSTALL_STATUS.md)를 참고한다.

## 유지할 요구사항

- 한 번에 한 명이 발화한다. 특정 사람의 목소리를 등록·인증하는 기능은 현재 범위에 포함하지 않는다.
- 먼저 ‘얼쑤’만 활성화한다. 기존 설정의 ‘좋다’는 이후 활성화할 수 있도록 보존한다.
- 독립적으로 외친 추임새에 반응한다. 단어를 설명하는 문장이나 일반 문장의 일부는 기본적으로 재생 대상에서 제외한다.
- 발화 종료부터 음원 재생 시작까지 4초 미만을 목표로 한다. 의도적인 추가 대기는 넣지 않는다.
- 음원 재생 중의 새 발화는 받지 않는 첫 버전으로 구현한다.
- 모델·음원 준비 → 입력 → 발화 구간 검출 → 음성 인식 → 추임새 판정 → 중복·지연 확인 → 음원 재생의 순서를 유지한다.

## 우선 비교할 인식 엔진

### 1. NAVER CLOVA Speech 실시간 스트리밍

공식 문서에서 한국어, gRPC 스트리밍, 키워드 부스팅, 발화 종료 기준 설정, 인식 결과 및 음절별 신뢰도·시간 정보를 확인했다. 입력은 16 kHz, 모노, 16-bit PCM이다.

이 프로그램에서 활용할 수 있는 점:

- 한국어 설정과 ‘얼쑤’의 키워드 부스팅을 시험할 수 있다.
- 인식 텍스트뿐 아니라 해당 음절의 시간·신뢰도를 판정에 활용할 수 있다.
- 스트리밍 연결로 오디오를 보내되, 재생은 추임새 판정이 완료된 결과에 대해서만 실행한다.
- 발화 종료 기준을 짧은 추임새에 맞게 조정할 수 있다. 공식 예제의 기본 대기값을 그대로 복사하지 않는다.

주의할 점: 부스팅이 강할수록 비슷한 단어가 잘못 인식될 수 있다. 부스팅 없음·약한 부스팅을 같은 녹음으로 비교해야 한다. 빈 인식 결과의 신뢰도 기본값이 1인 경우가 있으므로 텍스트·음성 구간 확인 없이 점수만 사용하면 안 된다. 계정, 스트리밍 도메인, API 인증, 네트워크와 사용 요금이 필요하다.

권고: 외부 API의 첫 시험 후보. 이것은 기능 적합성에 따른 권고이며 한국어 정확도 1위라는 의미가 아니다.

출처: [실시간 스트리밍 API](https://api.ncloud-docs.com/docs/ai-application-service-clovaspeech-grpc), [키워드 부스팅 사용 가이드](https://guide.ncloud-docs.com/docs/clovaspeech-builder-streaming).

### 2. MLX Whisper

기존 코드에 이미 연동되어 있으며 Apple Silicon에서 로컬로 실행할 수 있다. 현재 commands.json의 모델명은 openai/whisper-large-v3-turbo이므로 코드의 auto 규칙상 transformers 엔진이 선택된다. MLX를 사용하려면 MLX 형식의 모델과 엔진 설정을 맞춰야 한다.

권고: 외부 API와 비교할 로컬 기준 엔진. 인터넷 장애 시 선택 가능한 별도 실행 모드로 유지한다. API 응답을 오래 기다린 다음 같은 발화를 로컬에서 다시 처리하는 순차 재시도는 4초 목표에 불리하므로 기본 동작으로 넣지 않는다.

출처: [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [인식 결과와 세그먼트 처리 코드](https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/transcribe.py).

### 3. Google Cloud Speech-to-Text Chirp 3

한국어 ko-KR, V2 StreamingRecognize, 단어·문구 목록을 통한 adaptation을 지원한다. 공식 문서에는 단어별 confidence 반환값을 실제 신뢰도로 사용할 수 없다는 제한이 있다.

권고: CLOVA와 로컬 엔진으로 요구 성능을 얻지 못했을 때 비교할 후보. 엔드포인트의 지원 언어·리전·기능을 확인하고, 단어 신뢰도 필터를 전제로 구현하지 않는다.

출처: [Chirp 3 공식 문서](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3).

### 4. Azure Speech

한국어 인식과 실시간 인식용 phrase list를 제공한다. phrase list는 별도 모델 학습 없이 사용할 수 있고 가중치를 조정할 수 있다. 적용할 언어·엔드포인트의 지원 여부를 확인해 사용한다.

권고: 이미 Azure 계정을 사용하는 경우 유용한 비교 후보. 이번 프로젝트에서 가장 먼저 붙일 근거는 CLOVA보다 약하다. 이 판단 역시 실제 정확도 비교 결과가 아니다.

출처: [지원 언어](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt), [Phrase list](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/improve-accuracy-phrase-list).

### 5. OpenAI Realtime transcription

현재 공식 문서는 신규 실시간 전사에 gpt-live-transcribe를 권장한다. 오디오 입력 중 부분 전사를 받고, 클라이언트가 발화 종료를 판단해 commit하면 최종 전사를 받는 방식이다. 이 모델은 server_vad/semantic_vad를 지원하지 않으므로 클라이언트 측 발화 구간 검출을 유지해야 한다.

키워드·언어 힌트와 지연 수준을 설정할 수 있으나, 단어별 타임스탬프·전사 신뢰도 점수는 제공하지 않는다. 따라서 기존 시퀀스의 ‘품질 확인’을 신뢰도 임곗값 하나로 구현할 수 없다. 현재 안내의 모델명이 과거 예제와 다르므로 오래된 예제를 그대로 복사하지 않는다.

권고: 비교 가능한 API 후보. 이번 요구에서는 음절별 신뢰도·시간을 제공하는 CLOVA를 먼저 시험한다. 외부 API의 한국어 짧은 추임새 정확도는 모두 실제 녹음으로 확인해야 한다.

출처: [OpenAI Realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription), [전사 방식 안내](https://developers.openai.com/api/docs/guides/transcription).

## 전처리·단어 감지 라이브러리

### Silero VAD: 기본 도입 후보

음량만 보던 발화 검출을 말소리 확률 기반 검출로 보완한다. 16 kHz 입력과 ONNX 실행을 지원한다. VAD는 말소리 검출기이며, 발화자 분리나 주변 대화 제거 기능이 아니다. 짧고 강한 추임새를 놓치지 않도록 시작·종료 임곗값과 앞부분 보존 시간을 조정한다.

출처: [Silero VAD](https://github.com/snakers4/silero-vad).

### RNNoise: 선택적으로 비교

음성용 잡음 억제 라이브러리다. 잡음이 큰 환경에서 원본 입력과 적용한 입력을 비교한다. 기본 도입만으로 주변 대화가 제거된다고 가정하지 않는다. 추임새의 초성·늘인 발음이 손상되는지도 확인한 뒤 사용 여부를 정한다.

출처: [RNNoise](https://github.com/xiph/rnnoise).

### Porcupine: 보조 실험 후보

한국어 사용자 지정 키워드 모델과 macOS arm64 Python 실행을 지원한다. 계정·AccessKey와 사용자 지정 모델 생성이 필요하다. ‘얼쑤’ 모델의 실제 생성 가능 여부와 길게 외치는 발음의 성능은 별도 시험이 필요하다.

특정 단어 감지에 맞는 도구이지만, 합의한 음성 인식 시퀀스를 대체하지 않는다. 첫 버전에서 필수 관문으로 넣으면 이 단계의 누락이 전체 누락으로 이어질 수 있으므로 기본 경로에는 넣지 않는다. 이후 음성 인식 결과와 함께 비교하는 보조 신호로 시험할 수 있다. 일반 문장에 포함된 실제 단어도 감지할 수 있으므로 추임새 의도 판정을 대신할 수 없다.

출처: [사용자 지정 모델·한국어 지원](https://picovoice.ai/docs/model-api/porcupine/), [Python·Mac 지원 및 인증 요건](https://picovoice.ai/docs/quick-start/porcupine-python/).

openWakeWord는 현재 공식 README에서 영어만 지원한다고 명시하므로, 한국어 모델 학습이 필요한 첫 구현의 기본 후보에서 제외한다.

출처: [openWakeWord 언어 지원](https://github.com/dscripka/openWakeWord#language-support).

## 구현에 반영할 구조

1. 음원과 모델·API 연결을 준비한 뒤 준비 완료 상태를 표시한다.
2. 마이크 오디오에 캡처 시각을 붙이고 발화 앞부분을 보존한다.
3. Silero VAD로 발화 종료를 판단한다. 긴 대화는 최대 길이와 겹치는 구간을 두어 처리한다. 스트리밍 API에서는 연결 자체를 매 구간 재생성하지 않는다.
4. 실행 시작 시 선택한 엔진 하나로 인식한다. 공통 결과에는 텍스트, 오디오 구간, 완료 여부를 두고 신뢰도·음절 시간은 지원 엔진에서만 제공한다.
5. 최종 결과의 등록 추임새·주변 문맥·발화 길이·사용 가능한 엔진별 품질 정보를 확인한다. 부분 전사만 보고 즉시 재생하지 않는다. 구간을 나누며 일반 문장의 일부가 독립 추임새처럼 보이는 경우도 검증한다.
6. 캡처 시각에 근거해 지연을 확인하고, 같은 음성 구간의 중복 결과는 한 번만 처리한다. 끝이 없는 대기열을 두지 않는다. 과부하로 버린 입력과 기한 초과는 누락 통계에 남긴다.
7. 통과한 결과에 해당하는 음원을 한 번 재생한다. 재생 중에는 발화 판정을 중지하고, 그 시간대에 캡처된 오디오·미처리 결과도 재생 종료 후 다시 처리하지 않는다.

외부 API의 네트워크 왕복이나 종료 판단을 포함한 4초 상한은 문서만으로 보장할 수 없다. 늦은 결과의 재생을 막는 장치와, 정상 발화를 기한 내 인식하는 능력을 각각 평가해야 한다.

## 비교 검증

- 발화: ‘얼쑤’의 보통·작은·큰 목소리, 빠른 발음, 늘인 발음. 출력용 저장 음원만을 시험 입력으로 사용하지 않는다.
- 오작동: ‘어서’, ‘얼굴’, ‘얼씨구’, ‘얼쑤라고 말해 봐’, 일반 대화, 잡음·음악. 등록하지 않은 단어는 자동으로 동의어 처리하지 않는다.
- 환경: 조용한 방, 주변 대화, 음악·잡음, 실제 사용 위치와 마이크 거리. VAD·잡음 억제만으로 특정 사람의 말만 분리된다고 가정하지 않는다.
- 비교: 같은 라벨링된 녹음을 로컬 MLX와 CLOVA에서 각각 평가한다. 키워드 부스팅 없음·약한 부스팅도 비교한다. 초기 시험과 별도의 검증 녹음을 두어 설정에 맞춘 과대평가를 줄인다.
- 기록: 대상 추임새 감지율, 잘못 재생한 횟수/시간, 중복 재생, 기한 초과·과부하 누락, 발화 종료→실제 출력 시작의 중앙값·95백분위·최댓값, 4초 이상 지연된 비율.
- API 결과 수신 시각이나 재생 함수 호출 시각만으로 실제 출력 시작을 단정하지 않는다. 오디오 출력 시각 및 음원 앞부분의 무음을 확인하고, 최종 점검에서는 녹음으로 발화 종료와 출력 시작을 대조한다.

아직 실제 인식·성능 비교나 유료 API 호출은 수행하지 않았다. 우선 권고는 Silero VAD + CLOVA 스트리밍을 외부 후보로, Silero VAD + MLX Whisper를 로컬 비교 기준으로 구현할 수 있게 하는 것이다. 최종 엔진은 같은 환경의 측정 결과로 결정한다.


## 2026-11-13 목소리별 음원 분기

선택 모델은 [Common-Voice-Gender-Detection](https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection)이다. 모델 카드와 config.json에서 Wav2Vec2ForSequenceClassification 구조와 0=female, 1=male을 확인했다. [전처리 설정](https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection/blob/main/preprocessor_config.json)은 16 kHz, 입력 정규화를 지정한다. 모델은 영어로 태그되어 있으므로 공개 평가 수치를 한국어 추임새의 성능으로 사용하지 않는다.

[Transformers 공식 Wav2Vec2 문서](https://huggingface.co/docs/transformers/model_doc/wav2vec2)에 따라 특성 추출기와 분류 헤드를 함께 사용한다. 모델을 매 발화마다 로드하지 않고 기존 인식 worker에 상주시킨다. 모델은 CPU로 고정하고 Whisper의 Metal 실행 경로는 유지한다. 분류 시간도 기존 인식 소요시간과 10초 출력 기한 안에 포함된다.

음높이 방식 검토에서는 기존 남_얼쑤.wav의 음높이가 약 270 Hz로 추정되어 단순 임곗값이 잘못된 그룹을 선택했다. 최종 코드에는 음높이 임곗값 분류와 그 대체 경로를 포함하지 않는다. 다운로드 제한 때문에 실제 학습된 모델의 정확도를 아직 검사하지 못했다.
