"""추임새 판정과 발화 분할. 오디오 장치·MLX 없이 검증 가능한 로직."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 512
STEP = FRAME / SR
DEFAULTS = {
    'model': 'mlx-community/whisper-small-mlx',
    'word_timestamps': False,
    'language': 'ko', 'input_device': None, 'output_device': None,
    'vad_start': 0.6, 'vad_end': 0.35, 'start_frames': 2,
    'silence_duration': 0.32, 'pre_roll': 0.224,
    'min_speech': 0.128, 'max_speech': 2.4, 'overlap': 0.384,
    'max_latency': 10.0, 'output_margin': 0.05, 'echo_guard': 0.2,
    'max_frame_age': 0.25, 'cooldown': 0.7,
    'min_avg_logprob': -0.8, 'max_no_speech_prob': 0.55,
    'max_compression_ratio': 2.4, 'min_word_probability': 0.35,
    'event_log': 'logs/events.jsonl',
}


def normalize(text):
    # 문장부호/공백만 무시. 숫자·영문·다른 문자는 제거하여 오인하지 않는다.
    return ''.join(c for c in unicodedata.normalize('NFC', str(text)).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def resolve_path(base: Path, value: str):
    """macOS의 NFC/NFD 파일명 모두 지원. 실제 존재하는 경로를 우선한다."""
    raw = Path(value).expanduser()
    raw = raw if raw.is_absolute() else base / raw
    for form in (None, 'NFC', 'NFD'):
        path = raw if form is None else Path(unicodedata.normalize(form, str(raw)))
        if path.exists():
            return path.resolve()
    return raw


@dataclass(frozen=True)
class Command:
    name: str
    phrases: tuple[str, ...]
    sounds: tuple[Path, ...]
    volume: float = 1.0
    cooldown: float = 0.7


@dataclass(frozen=True)
class Config:
    path: Path
    settings: dict
    commands: tuple[Command, ...]

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('설정 파일의 최상위 값은 JSON 객체여야 합니다')
        settings = dict(DEFAULTS)
        provided = data.get('settings', {})
        if not isinstance(provided, dict):
            raise ValueError('settings는 JSON 객체여야 합니다')
        unknown = set(provided) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'알 수 없는 설정: {sorted(unknown)}')
        settings.update(provided)
        for key, default in DEFAULTS.items():
            if isinstance(default, bool):
                if not isinstance(settings[key], bool):
                    raise ValueError(f'{key}: true 또는 false여야 합니다')
            elif isinstance(default, (int, float)):
                value = settings[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{key}: 유한한 숫자가 필요합니다')
        if not 0 < settings['vad_end'] < settings['vad_start'] < 1:
            raise ValueError('0 < vad_end < vad_start < 1 이어야 합니다')
        if not isinstance(settings['start_frames'], int) or not 1 <= settings['start_frames'] <= 10:
            raise ValueError('start_frames는 1~10 사이 정수여야 합니다')
        if not STEP <= settings['silence_duration'] <= 1:
            raise ValueError('silence_duration은 0.032~1초여야 합니다')
        if not 0 < settings['min_speech'] < settings['max_speech'] <= 5:
            raise ValueError('0 < min_speech < max_speech <= 5 이어야 합니다')
        if not 0 <= settings['overlap'] < settings['max_speech'] / 2:
            raise ValueError('overlap은 max_speech의 절반보다 작아야 합니다')
        if not 0 <= settings['pre_roll'] <= 1:
            raise ValueError('pre_roll은 0~1초여야 합니다')
        if not 0 < settings['max_latency'] <= 10:
            raise ValueError('max_latency는 0초 초과, 10초 이하여야 합니다')
        if not 0 <= settings['echo_guard'] <= 2 or not 0 <= settings['cooldown'] <= 10:
            raise ValueError('echo_guard 또는 cooldown 범위 오류')
        if not 0.02 <= settings['output_margin'] < settings['max_latency']:
            raise ValueError('output_margin은 0.02초 이상이고 max_latency보다 작아야 합니다')
        if not 0 < settings['max_frame_age'] <= 1:
            raise ValueError('max_frame_age는 0초 초과, 1초 이하여야 합니다')
        if not -5 <= settings['min_avg_logprob'] <= 0:
            raise ValueError('min_avg_logprob 범위 오류')
        for key in ('max_no_speech_prob', 'min_word_probability'):
            if not 0 <= settings[key] <= 1:
                raise ValueError(f'{key}는 0~1 사이여야 합니다')
        if settings['max_compression_ratio'] <= 0:
            raise ValueError('max_compression_ratio는 양수여야 합니다')
        if not isinstance(settings['model'], str) or not settings['model'].strip():
            raise ValueError('model이 필요합니다')
        if settings['language'] != 'ko':
            raise ValueError('현재 프로그램은 language=ko만 지원합니다')
        if settings['event_log'] is not None and not isinstance(settings['event_log'], str):
            raise ValueError('event_log는 문자열 또는 null이어야 합니다')
        for key in ('input_device', 'output_device'):
            value = settings[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
                raise ValueError(f'{key}는 장치 이름/번호 또는 null이어야 합니다')
        commands, seen, names = [], set(), set()
        entries = data.get('commands', [])
        if not isinstance(entries, list):
            raise ValueError('commands는 목록이어야 합니다')
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError('각 명령은 JSON 객체여야 합니다')
            if not isinstance(entry.get('enabled', True), bool):
                raise ValueError('enabled는 true/false여야 합니다')
            if not entry.get('enabled', True):
                continue
            if entry.get('match', 'exact') != 'exact':
                raise ValueError('오작동 방지를 위해 match=exact만 지원합니다')
            name = entry.get('name')
            phrases = entry.get('phrases', [])
            sounds = entry.get('sounds', [])
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError('명령 이름이 비어 있거나 중복됩니다')
            if not isinstance(phrases, list) or not phrases or not all(isinstance(p, str) for p in phrases):
                raise ValueError(f'{name}: phrases에 문자열 목록이 필요합니다')
            phrases = tuple(normalize(p) for p in phrases)
            if any(not p or p in seen for p in phrases) or len(set(phrases)) != len(phrases):
                raise ValueError(f'{name}: phrases가 비어 있거나 중복됩니다')
            if isinstance(sounds, str):
                sounds = [sounds]
            if not isinstance(sounds, list) or not sounds or not all(isinstance(p, str) and p for p in sounds):
                raise ValueError(f'{name}: sounds에 파일 목록이 필요합니다')
            files = tuple(resolve_path(path.parent, p) for p in sounds)
            if any(not p.is_file() for p in files):
                raise ValueError(f'{name}: 음원이 없습니다: {files}')
            volume = entry.get('volume', 1.0)
            cooldown = entry.get('cooldown', settings['cooldown'])
            for value in (volume, cooldown):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{name}: volume/cooldown 숫자 오류')
            if not 0 < volume <= 2 or not 0 <= cooldown <= 10:
                raise ValueError(f'{name}: volume/cooldown 범위 오류')
            commands.append(Command(name, phrases, files, volume, cooldown))
            seen.update(phrases)
            names.add(name)
        if not commands:
            raise ValueError('활성 명령이 하나 이상 필요합니다')
        if data.get('fallback') is not None:
            raise ValueError('등록하지 않은 발화에는 음원을 출력하지 않습니다. fallback은 null로 설정하세요')
        return cls(path, settings, tuple(commands))


@dataclass
class Utterance:
    audio: np.ndarray
    start: float
    speech_end: float
    voiced_seconds: float
    continued: bool = False
    truncated: bool = False
    uid: int = 0


class Segmenter:
    """512-sample 프레임과 캡처 시각을 사용한다. 처리 시각으로 발화 끝을 덮지 않는다.

    긴 대화에서 분리된 마지막 단어를 독립 추임새로 오인하지 않도록,
    최대 길이로 잘린 구간과 그 연속 구간은 모두 continued/truncated로 표시한다.
    """
    def __init__(self, settings):
        self.s = settings
        self.serial = 0
        self.reset()

    def reset(self):
        self.pre = deque(maxlen=max(self.s['start_frames'], math.ceil(self.s['pre_roll'] / STEP)))
        self.frames = []
        self.active = False
        self.starts = 0
        self.silent = 0
        self.voiced = 0
        self.continued = False
        self.last_end = None
        self.speech_end = 0.0

    def feed(self, frame, end, probability):
        if len(frame) != FRAME or not np.isfinite(frame).all():
            raise ValueError('VAD 프레임은 유한한 512개 샘플이어야 합니다')
        if not math.isfinite(end) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('잘못된 프레임 시각 또는 VAD 확률')
        if self.last_end is not None and abs(end - self.last_end - STEP) > STEP / 2:
            self.reset()
        self.last_end = end
        if not self.active:
            self.pre.append((frame.copy(), end))
            self.starts = self.starts + 1 if probability >= self.s['vad_start'] else 0
            if self.starts >= self.s['start_frames']:
                self.active = True
                self.frames = list(self.pre)
                self.voiced = self.starts
                self.speech_end = end
            return None
        self.frames.append((frame.copy(), end))
        if probability >= self.s['vad_end']:
            self.silent = 0
            self.voiced += 1
            self.speech_end = end
        else:
            self.silent += 1
        ended = self.silent * STEP >= self.s['silence_duration']
        truncated = len(self.frames) * STEP >= self.s['max_speech']
        if not ended and not truncated:
            return None
        item = self._emit(truncated and not ended)
        if ended:
            self.reset()
        else:
            keep = max(1, math.ceil(self.s['overlap'] / STEP))
            self.frames = self.frames[-keep:]
            self.continued = True
            self.voiced = min(self.voiced, keep)
        return item

    def _emit(self, truncated):
        if self.voiced * STEP < self.s['min_speech']:
            return None
        self.serial += 1
        return Utterance(np.concatenate([x for x, _ in self.frames]),
                         self.frames[0][1] - STEP, self.speech_end,
                         self.voiced * STEP, self.continued, truncated, self.serial)

    def flush(self):
        # 파일 끝에서만 사용한다. 실시간의 강제 종료에서는 미완료 발화를 재생하지 않는다.
        item = self._emit(False) if self.active else None
        self.reset()
        return item


def text_command(text, commands):
    value = normalize(text)
    return next((c for c in commands if value in c.phrases), None) if value else None


def evaluate(result, utterance, cfg):
    """Whisper의 점수는 정답 확률이 아니다. 보수적인 품질 신호로만 사용한다."""
    if utterance.truncated or utterance.continued:
        return None, 'continuous_speech'
    text = result.get('text', '')
    command = text_command(text, cfg.commands)
    if command is None:
        return None, 'not_registered_interjection'
    segments = result.get('segments') or []
    spoken = [s for s in segments if normalize(s.get('text', ''))]
    if not spoken:
        return None, 'missing_quality_metadata'
    s = cfg.settings
    for segment in spoken:
        for key in ('avg_logprob', 'no_speech_prob', 'compression_ratio'):
            value = segment.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return None, 'missing_quality_metadata'
        if segment['avg_logprob'] < s['min_avg_logprob']:
            return None, 'low_logprob'
        if segment['no_speech_prob'] > s['max_no_speech_prob']:
            return None, 'no_speech'
        if segment['compression_ratio'] > s['max_compression_ratio']:
            return None, 'repetition'
        if not s['word_timestamps']:
            continue
        words = [w for w in (segment.get('words') or []) if normalize(w.get('word', ''))]
        if not words:
            return None, 'missing_word_metadata'
        for word in words:
            probability = word.get('probability')
            if not isinstance(probability, (int, float)) or not math.isfinite(probability):
                return None, 'missing_word_metadata'
            if probability < s['min_word_probability']:
                return None, 'low_word_probability'
    return command, 'accepted'


class PlaybackGate:
    def __init__(self, settings):
        self.s = settings
        self.discard_before = -math.inf
        self.last_played = {}
        self.last_uid = -1

    def reason(self, item, command, now):
        if item.start < self.discard_before:
            return 'playback_overlap'
        if item.uid <= self.last_uid:
            return 'duplicate'
        if now >= item.speech_end + self.s['max_latency'] - self.s['output_margin']:
            return 'deadline'
        if now - self.last_played.get(command.name, -math.inf) < command.cooldown:
            return 'cooldown'
        return None

    def commit(self, item, command, now, blocked_until):
        self.last_uid = item.uid
        self.last_played[command.name] = now
        self.discard_before = blocked_until
