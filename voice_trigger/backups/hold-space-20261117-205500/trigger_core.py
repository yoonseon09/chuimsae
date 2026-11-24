"""추임새 판정과 발화 분할. 오디오 장치·MLX 없이 검증 가능한 로직."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field, replace
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
    'min_speech': 0.128, 'max_speech': 5.0, 'overlap': 0.384,
    'max_latency': 10.0, 'output_margin': 0.05, 'echo_guard': 0.2,
    'max_frame_age': 0.25, 'cooldown': 0.7,
    'duplicate_window': 5.0, 'playback_gap': 2.0, 'burst_silence': 0.6,
    'queue_timeout': 60.0, 'queue_capacity': 16,
    'min_avg_logprob': -0.8, 'max_no_speech_prob': 0.55,
    'max_compression_ratio': 2.4, 'min_word_probability': 0.35,
    'event_log': 'logs/events.jsonl',
    'voice_mode': 'off',
    'voice_min_seconds': 0.16, 'voice_min_confidence': 0.8,
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
            resolved = path.resolve()
            # macOS는 NFC 조회도 허용하지만 실제 이름은 NFD일 수 있다.
            # glob 결과와 같은 경로 표기를 반환해 음원 캐시 키를 맞춘다.
            for entry in resolved.parent.iterdir():
                if unicodedata.normalize('NFC', entry.name) == unicodedata.normalize('NFC', resolved.name):
                    return entry
            return resolved
    return raw


def voice_files(base, patterns):
    """실행 시작 시 음원 패턴 확장. NFC/NFD와 파일 추가 전 빈 그룹을 지원한다."""
    files = set()
    for pattern in patterns:
        if Path(pattern).is_absolute() or '..' in Path(pattern).parts:
            raise ValueError('voice_sounds는 설정 폴더 아래 상대 경로를 사용하세요')
        for form in ('NFC', 'NFD'):
            files.update(p.resolve() for p in base.glob(unicodedata.normalize(form, pattern)) if p.is_file())
    return tuple(sorted(files))


@dataclass(frozen=True)
class Command:
    name: str
    phrases: tuple[str, ...]
    sounds: tuple[Path, ...]
    volume: float = 1.0
    cooldown: float = 0.7
    voice_sounds: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    match: str = 'word'

    @property
    def all_sounds(self):
        return tuple(dict.fromkeys((*self.sounds, *(p for group in self.voice_sounds.values() for p in group))))


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
        for key, low, high in [('duplicate_window', 0, 60), ('playback_gap', 0, 10),
                               ('burst_silence', 0.32, 2), ('queue_timeout', 1, 300)]:
            if not low <= settings[key] <= high:
                raise ValueError(f'{key}는 {low}~{high}초여야 합니다')
        if not isinstance(settings['queue_capacity'], int) or not 1 <= settings['queue_capacity'] <= 64:
            raise ValueError('queue_capacity는 1~64 사이 정수여야 합니다')
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
        if settings['voice_mode'] not in ('off', 'model'):
            raise ValueError('voice_mode는 off 또는 model이어야 합니다')
        if not 0.08 <= settings['voice_min_seconds'] <= 2:
            raise ValueError('voice_min_seconds는 0.08~2초여야 합니다')
        if not 0.5 < settings['voice_min_confidence'] <= 1:
            raise ValueError('voice_min_confidence는 0.5 초과, 1 이하여야 합니다')
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
            if entry.get('match', 'word') not in ('word', 'exact'):
                raise ValueError('match는 word 또는 exact여야 합니다')
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
            groups = entry.get('voice_sounds', {})
            if not isinstance(groups, dict) or set(groups) - {'male', 'female'}:
                raise ValueError(f'{name}: voice_sounds에는 male/female만 지정할 수 있습니다')
            routed = {}
            for label, patterns in groups.items():
                if not isinstance(patterns, list) or not all(isinstance(p, str) and p for p in patterns):
                    raise ValueError(f'{name}: voice_sounds.{label}은 파일 경로/패턴 목록이어야 합니다')
                routed[label] = voice_files(path.parent, patterns)
            commands.append(Command(name, phrases, files, volume, cooldown, routed,
                                    entry.get('match', 'word')))
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

    긴 발화는 겹치는 구간을 두고 분할한다. 문장 안의 추임새도 처리하므로
    continued/truncated 구간도 인식하며, 겹친 추임새는 출력 대기열에서 합친다.
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
    """단일 결과가 필요한 기존 호출자를 위한 첫 번째 일치."""
    return next(iter(text_commands(text, commands)), None)


def text_commands(text, commands):
    """문장 안의 독립 단어를 발화 순서대로 찾고, 같은 명령은 한 번만 반환한다.

    '얼쑤라고'처럼 더 긴 단어의 일부는 제외한다. 붙여 쓴 추임새 반복은
    같은 단어의 반복으로 취급한다(얼쑤얼쑤얼쑤). 유사 문자열 추정은 하지 않는다.
    """
    value = unicodedata.normalize('NFC', str(text)).casefold()
    found = []
    for command in commands:
        # 기존 전체 일치의 공백·문장부호 처리도 유지한다.
        if normalize(value) in command.phrases:
            found.append((0, command))
            continue
        if command.match == 'exact':
            continue
        aliases = '|'.join('\\s*'.join(re.escape(c) for c in phrase)
                           for phrase in sorted(command.phrases, key=len, reverse=True))
        match = re.search(r'(?<!\w)(?:' + aliases + r')+(?!\w)', value)
        if match:
            found.append((match.start(), command))
    return [command for _, command in sorted(found, key=lambda pair: pair[0])]


def evaluate_commands(result, utterance, cfg):
    """Whisper의 점수는 정답 확률이 아니다. 보수적인 품질 신호로만 사용한다."""
    text = result.get('text', '')
    commands = text_commands(text, cfg.commands)
    if not commands:
        return [], 'not_registered_interjection'
    segments = result.get('segments') or []
    spoken = [s for s in segments if normalize(s.get('text', ''))]
    if not spoken:
        return [], 'missing_quality_metadata'
    s = cfg.settings
    for segment in spoken:
        for key in ('avg_logprob', 'no_speech_prob', 'compression_ratio'):
            value = segment.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return [], 'missing_quality_metadata'
        if segment['avg_logprob'] < s['min_avg_logprob']:
            return [], 'low_logprob'
        if segment['no_speech_prob'] > s['max_no_speech_prob']:
            return [], 'no_speech'
        if segment['compression_ratio'] > s['max_compression_ratio']:
            return [], 'repetition'
        if not s['word_timestamps']:
            continue
        words = [w for w in (segment.get('words') or []) if normalize(w.get('word', ''))]
        if not words:
            return [], 'missing_word_metadata'
        for word in words:
            probability = word.get('probability')
            if not isinstance(probability, (int, float)) or not math.isfinite(probability):
                return [], 'missing_word_metadata'
            if probability < s['min_word_probability']:
                return [], 'low_word_probability'
    return commands, 'accepted'


def evaluate(result, utterance, cfg):
    commands, reason = evaluate_commands(result, utterance, cfg)
    return next(iter(commands), None), reason


@dataclass(frozen=True)
class PlaybackRequest:
    item: Utterance
    command: Command
    deadline: float


class PlaybackQueue:
    """인식된 후보의 순서를 보존한다. 중복은 처리 시각 대신 입력 시각으로 판단한다."""
    def __init__(self, settings):
        self.s = settings
        self.pending = deque()
        self.last_seen = {}
        self.last_uid = {}

    def offer(self, item, command, now):
        if now >= item.speech_end + self.s['max_latency'] - self.s['output_margin']:
            return 'deadline'
        if item.uid <= self.last_uid.get(command.name, -1):
            return 'duplicate'
        previous = self.last_seen.get(command.name, -math.inf)
        duplicate = (any(r.command.name == command.name for r in self.pending)
                     or item.start - previous < max(self.s['duplicate_window'], command.cooldown))
        if duplicate:
            # 연속 입력이 이어지는 동안 억제 시간을 연장한다.
            self.last_uid[command.name] = item.uid
            self.last_seen[command.name] = max(previous, item.speech_end)
            return 'repeated_interjection'
        if len(self.pending) >= self.s['queue_capacity']:
            return 'playback_queue_full'
        self.last_uid[command.name] = item.uid
        self.last_seen[command.name] = item.speech_end
        self.pending.append(PlaybackRequest(item, command, now + self.s['queue_timeout']))
        return None

    def expire(self, now):
        expired = []
        while self.pending and now >= self.pending[0].deadline:
            expired.append(self.pending.popleft())
        return expired


def route_command(command, voice, settings):
    """판정 불가/해당 음원 없음은 다른 목소리의 음원으로 대체하지 않는다."""
    if settings['voice_mode'] == 'off':
        return command, 'accepted'
    label = voice.get('label') if isinstance(voice, dict) else None
    if label not in ('male', 'female'):
        return None, 'voice_unknown'
    sounds = command.voice_sounds.get(label, ())
    if not sounds:
        return None, 'voice_sound_missing'
    return replace(command, sounds=sounds), 'accepted'

