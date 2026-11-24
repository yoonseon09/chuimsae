#!/usr/bin/env python3
"""
voice_trigger.py — 한국어 음성인식 → 인식된 말에 맞는 음원 재생

동작 흐름
  1) 마이크를 계속 듣다가 말소리가 시작되면 녹음, 말이 끝나면(무음 감지) 한 문장으로 자름
  2) Whisper(mlx-whisper, Apple Silicon 최적화)로 한국어 텍스트 변환
  3) commands.json 의 phrases 와 비교해 가장 잘 맞는 명령을 찾음
  4) 그 명령의 sounds 중 하나를 출력 장치로 재생

commands.json 은 실행 중에 고쳐도 자동으로 다시 읽습니다(저장만 하면 됨).

사용법
  python voice_trigger.py                  # 마이크로 실행
  python voice_trigger.py --list-devices   # 오디오 장치 번호/이름 확인
  python voice_trigger.py --type           # 마이크 대신 키보드로 문장 입력해 매칭 테스트
  python voice_trigger.py --test "안녕하세요"  # 한 문장만 매칭 테스트
"""

import argparse
import json
import os
import queue
import random
import re
import sys
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np

try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # rapidfuzz 가 없으면 표준 라이브러리로 대체
    _rf_fuzz = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "commands.json")
WHISPER_SR = 16000
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff")

DEFAULT_SETTINGS = {
    "model": "mlx-community/whisper-large-v3-turbo",
    "engine": "auto",
    "language": "ko",
    "input_device": None,
    "output_device": None,
    "start_threshold_db": -40,
    "silence_duration": 0.4,
    "min_speech": 0.2,
    "max_speech": 4.0,
    "pre_roll": 0.3,
    "match_threshold": 75,
    "play_mode": "interrupt",
    "ignore_while_playing": True,
    "use_phrases_as_prompt": True,
    "ignore_texts": [],
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────── 설정 파일 ───────────────────────────────

class Config:
    """commands.json 을 읽고, 파일이 바뀌면 자동으로 다시 읽는다."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.base_dir = os.path.dirname(self.path)
        self._mtime = None
        self.settings = dict(DEFAULT_SETTINGS)
        self.commands = []
        self.fallback = None
        self.reload(force=True)

    def reload(self, force=False):
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            if force:
                sys.exit(f"설정 파일을 찾을 수 없습니다: {self.path}")
            return False
        if not force and mtime == self._mtime:
            return False
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"commands.json 형식 오류 (줄 {e.lineno}, 칸 {e.colno}): {e.msg}"
            if force:
                sys.exit(msg)
            log(msg + " → 이전 설정을 계속 사용합니다.")
            self._mtime = mtime
            return False

        settings = dict(DEFAULT_SETTINGS)
        settings.update(data.get("settings", {}))
        commands = []
        for i, c in enumerate(data.get("commands", [])):
            phrases = [p for p in c.get("phrases", []) if str(p).strip()]
            sounds = c.get("sounds", [])
            if isinstance(sounds, str):
                sounds = [sounds]
            if not phrases or not sounds:
                log(f"commands[{i}] '{c.get('name', '')}' 는 phrases 또는 sounds 가 비어 있어 건너뜁니다.")
                continue
            commands.append({
                "name": c.get("name", phrases[0]),
                "phrases": phrases,
                "_norm": [normalize(p) for p in phrases],
                "sounds": [self.resolve(s) for s in sounds],
                "volume": float(c.get("volume", 1.0)),
                "cooldown": float(c.get("cooldown", 0)),
                "match": c.get("match", "auto"),
            })
        fb = data.get("fallback")
        if fb and fb.get("sounds"):
            fs = fb["sounds"] if isinstance(fb["sounds"], list) else [fb["sounds"]]
            fb = {"name": "fallback", "sounds": [self.resolve(s) for s in fs],
                  "volume": float(fb.get("volume", 1.0)), "cooldown": 0}
        else:
            fb = None

        for c in commands + ([fb] if fb else []):
            for s in c["sounds"]:
                if os.path.isdir(s):
                    if not list_audio(s):
                        log(f"⚠️  폴더에 음원 파일이 없음: {s}")
                elif not os.path.isfile(s):
                    log(f"⚠️  음원 파일 없음: {s}")

        self.settings, self.commands, self.fallback = settings, commands, fb
        self._mtime = mtime
        if not force:
            log(f"🔄 commands.json 다시 읽음 (명령 {len(commands)}개)")
        return True

    def resolve(self, p):
        p = os.path.expanduser(unicodedata.normalize("NFC", str(p)))
        return p if os.path.isabs(p) else os.path.join(self.base_dir, p)


def list_audio(folder):
    """폴더 안의 음원 파일 목록 (숨김 파일 제외, 이름순)."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names
            if not n.startswith(".") and n.lower().endswith(AUDIO_EXTS)
            and os.path.isfile(os.path.join(folder, n))]


def expand_sounds(sounds):
    """sounds 목록의 폴더를 그 안의 파일들로 펼친다. 폴더는 매번 다시 읽어 새 파일도 바로 반영."""
    files = []
    for s in sounds:
        files.extend(list_audio(s) if os.path.isdir(s) else [s])
    return files


# ─────────────────────────────── 텍스트 매칭 ───────────────────────────────

_NON_WORD = re.compile(r"[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ]+")


def normalize(text):
    """비교용으로 한글을 표준형(NFC)으로 맞추고, 소문자화하고, 공백·문장부호를 모두 제거."""
    return _NON_WORD.sub("", unicodedata.normalize("NFC", str(text)).lower())


def _partial_ratio(a, b):
    """짧은 쪽(a)이 긴 쪽(b) 어딘가와 얼마나 비슷한지 0~100."""
    if _rf_fuzz is not None:
        return _rf_fuzz.partial_ratio(a, b)
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    for i in range(len(b) - len(a) + 1):
        best = max(best, SequenceMatcher(None, a, b[i:i + len(a)]).ratio())
        if best == 1.0:
            break
    return best * 100


def _ratio(a, b):
    if _rf_fuzz is not None:
        return _rf_fuzz.ratio(a, b)
    return SequenceMatcher(None, a, b).ratio() * 100


def find_match(text, commands, threshold):
    """
    인식된 문장과 가장 잘 맞는 명령을 찾는다.
    match 모드
      "auto"    : 문장 안에 phrase 가 들어 있거나 비슷하면 매칭 (기본)
      "exact"   : 문장 전체가 phrase 와 같을 때만
      "contains": phrase 가 글자 그대로 들어 있을 때만
    반환: (명령, 점수, 걸린 phrase) 또는 (None, 최고점수, None)
    """
    t = normalize(text)
    if not t:
        return None, 0, None
    best = (None, 0.0, None, 0)  # cmd, score, phrase, phrase_len
    for cmd in commands:
        mode = cmd.get("match", "auto")
        for phrase, p in zip(cmd["phrases"], cmd["_norm"]):
            if not p:
                continue
            if mode == "exact":
                score = 100.0 if p == t else 0.0
            elif mode == "contains":
                score = 100.0 if p in t else 0.0
            else:
                if p in t:
                    score = 100.0
                elif len(p) <= 2:
                    # 한두 글자짜리는 비슷한 글자만으로 오작동하기 쉬워서 정확히 들어 있을 때만
                    score = 0.0
                else:
                    score = max(_partial_ratio(p, t), _ratio(p, t))
            # 점수가 같으면 더 긴(구체적인) phrase 를 우선
            if score > best[1] or (score == best[1] and score > 0 and len(p) > best[3]):
                best = (cmd, score, phrase, len(p))
    cmd, score, phrase, _ = best
    if cmd is not None and score >= threshold:
        return cmd, score, phrase
    return None, score, None


# ─────────────────────────────── 오디오 재생 ───────────────────────────────

class Player:
    def __init__(self, cfg, dry_run=False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.busy_until = 0.0
        self.last_played = {}
        self.last_file = {}
        self._cache = {}

    def preload(self):
        """모든 음원을 미리 메모리에 올려 재생 지연을 없앤다."""
        n = 0
        for c in self.cfg.commands + ([self.cfg.fallback] if self.cfg.fallback else []):
            for path in expand_sounds(c["sounds"]):
                if os.path.isfile(path):
                    try:
                        self._load(path)
                        n += 1
                    except Exception as e:
                        log(f"❌ 음원을 읽을 수 없음 ({path}): {e}")
        log(f"📦 음원 {n}개 미리 불러옴")

    def _load(self, path):
        import soundfile as sf
        mtime = os.path.getmtime(path)
        hit = self._cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1], hit[2]
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        self._cache[path] = (mtime, data, sr)
        return data, sr

    def is_playing(self):
        return time.monotonic() < self.busy_until

    def play(self, cmd):
        now = time.monotonic()
        name = cmd["name"]
        if cmd.get("cooldown", 0) and now - self.last_played.get(name, -1e9) < cmd["cooldown"]:
            log(f"⏳ '{name}' 쿨다운 중이라 건너뜀")
            return False
        if self.cfg.settings.get("play_mode") == "skip_if_playing" and self.is_playing():
            log("⏭️  재생 중이라 건너뜀 (play_mode=skip_if_playing)")
            return False

        files = [f for f in expand_sounds(cmd["sounds"]) if os.path.isfile(f)]
        if not files:
            log(f"❌ '{name}' 에 재생할 음원이 없습니다: {cmd['sounds']}")
            return False
        # 파일이 여러 개면 직전에 나온 것은 피해서 무작위 선택
        choices = [f for f in files if f != self.last_file.get(name)] or files
        path = random.choice(choices)
        self.last_file[name] = path
        if self.dry_run:
            log(f"🔈 (재생 생략) {os.path.relpath(path, self.cfg.base_dir)}")
            self.last_played[name] = now
            return True
        try:
            import sounddevice as sd
            data, sr = self._load(path)
            vol = cmd.get("volume", 1.0)
            if vol != 1.0:
                data = np.clip(data * vol, -1.0, 1.0)
            device = resolve_device(self.cfg.settings.get("output_device"), "output")
            sd.play(data, sr, device=device)  # 이전 재생은 자동으로 멈춤
            self.busy_until = now + len(data) / sr + 0.25
            self.last_played[name] = now
            log(f"🔊 재생: {os.path.relpath(path, self.cfg.base_dir)} ({len(data) / sr:.1f}초)")
            return True
        except Exception as e:
            log(f"❌ 재생 실패 ({path}): {e}")
            return False


# ─────────────────────────────── 장치 선택 ───────────────────────────────

def resolve_device(spec, kind):
    """None(기본 장치) / 숫자(장치 번호) / 문자열(이름 일부) 을 장치 번호로 변환."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    import sounddevice as sd
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and str(spec).lower() in d["name"].lower():
            return i
    raise SystemExit(f"{'입력' if kind == 'input' else '출력'} 장치 '{spec}' 를 찾지 못했습니다. "
                     f"--list-devices 로 이름을 확인하세요.")


def list_devices():
    import sounddevice as sd
    din, dout = sd.default.device
    print("번호  입력  출력  이름")
    for i, d in enumerate(sd.query_devices()):
        mark = (" ← 기본 입력" if i == din else "") + (" ← 기본 출력" if i == dout else "")
        print(f"{i:>3}   {d['max_input_channels']:>3}   {d['max_output_channels']:>3}   {d['name']}{mark}")
    print('\ncommands.json 의 "input_device"/"output_device" 에 번호나 이름 일부를 적으세요.')


# ─────────────────────────────── 음성인식 ───────────────────────────────

def pick_engine(model, engine="auto"):
    """
    engine
      "mlx"          : mlx-whisper (Apple Silicon 전용, 가장 빠름). 예) mlx-community/whisper-large-v3-turbo
      "transformers" : Hugging Face 원본 모델 그대로.          예) openai/whisper-large-v3-turbo
      "auto"         : 모델 이름에 'mlx' 가 있으면 mlx, 아니면 transformers
    """
    engine = (engine or "auto").lower()
    if engine == "auto":
        return "mlx" if "mlx" in str(model).lower() else "transformers"
    if engine not in ("mlx", "transformers"):
        sys.exit(f'settings.engine 은 "auto", "mlx", "transformers" 중 하나여야 합니다 (지금: {engine})')
    return engine


class Recognizer:
    """설정의 model/engine 은 시작할 때 한 번 읽습니다. 바꾸면 프로그램을 다시 실행하세요."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.settings["model"]
        self.engine = pick_engine(self.model, cfg.settings.get("engine", "auto"))
        if self.engine == "mlx":
            try:
                import mlx_whisper
            except ImportError:
                sys.exit("mlx-whisper 가 설치되어 있지 않습니다.\n"
                         "  가상환경을 켠 상태에서:  python -m pip install mlx-whisper\n"
                         f"  (지금 실행 중인 파이썬: {sys.executable})")
            self._mw = mlx_whisper
        else:
            self._load_transformers()

    def _load_transformers(self):
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            sys.exit("transformers 엔진에는 transformers 와 torch 가 필요합니다.\n"
                     "  가상환경을 켠 상태에서:  python -m pip install transformers torch\n"
                     f"  (지금 실행 중인 파이썬: {sys.executable})")
        self._torch = torch
        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16      # M1 GPU
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        log(f"🧠 원본 모델 불러오는 중: {self.model} ({device}) — 처음엔 다운로드로 몇 분 걸립니다")
        self._pipe = pipeline("automatic-speech-recognition", model=self.model,
                              dtype=dtype, device=device)
        self._prompt_cache = (None, None)

    def warmup(self):
        log(f"🧠 모델 준비 중: {self.model} [엔진: {self.engine}] (처음 한 번은 다운로드로 시간이 걸립니다)")
        t0 = time.time()
        self.transcribe(np.zeros(WHISPER_SR // 2, dtype=np.float32))
        log(f"✅ 모델 준비 완료 ({time.time() - t0:.1f}초)")

    def _prompt_text(self):
        if not self.cfg.settings.get("use_phrases_as_prompt"):
            return None
        words = [p for c in self.cfg.commands for p in c["phrases"]]
        return ", ".join(dict.fromkeys(words))[:400] if words else None

    def transcribe(self, audio):
        audio = audio.astype(np.float32)
        if self.engine == "mlx":
            return self._transcribe_mlx(audio)
        return self._transcribe_transformers(audio)

    def _transcribe_mlx(self, audio):
        s = self.cfg.settings
        kwargs = dict(
            path_or_hf_repo=self.model,
            language=s.get("language") or None,
            condition_on_previous_text=False,
            temperature=0.0,
            verbose=None,
        )
        prompt = self._prompt_text()
        if prompt:
            kwargs["initial_prompt"] = prompt
        result = self._mw.transcribe(audio, **kwargs)
        text = (result.get("text") or "").strip()
        segs = result.get("segments") or []
        if segs and all(seg.get("no_speech_prob", 0) > 0.6 for seg in segs):
            return ""
        return text

    def _transcribe_transformers(self, audio):
        s = self.cfg.settings
        gen = {"task": "transcribe"}
        if s.get("language"):
            gen["language"] = s["language"]
        prompt = self._prompt_text()
        if prompt:
            # 프롬프트(등록 단어 힌트)는 토큰으로 바꿔 두고, 단어 목록이 바뀔 때만 다시 만든다
            if self._prompt_cache[0] != prompt:
                ids = self._pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
                self._prompt_cache = (prompt, ids.to(self._pipe.device))
            gen["prompt_ids"] = self._prompt_cache[1]
        with self._torch.inference_mode():
            out = self._pipe({"raw": audio, "sampling_rate": WHISPER_SR}, generate_kwargs=gen)
        text = (out.get("text") or "").strip()
        # 일부 버전은 결과 앞에 프롬프트를 붙여 돌려주므로 제거
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text


def is_ignored(text, ignore_list):
    t = normalize(text)
    if not t:
        return True
    return any(normalize(x) and normalize(x) in t and len(t) <= len(normalize(x)) + 4
               for x in ignore_list)


# ─────────────────────────────── 마이크 / 무음 감지 ───────────────────────────────

def to_16k(x, sr):
    if sr == WHISPER_SR:
        return x.astype(np.float32)
    n = int(round(len(x) * WHISPER_SR / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def run_mic(cfg, player, recognizer):
    import sounddevice as sd

    s = cfg.settings
    in_dev = resolve_device(s.get("input_device"), "input")
    info = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0])
    sr = int(info["default_samplerate"])
    block = int(sr * 0.03)  # 30ms 단위로 판정
    frames = queue.Queue()
    utterances = queue.Queue()

    def callback(indata, n, t, status):
        frames.put(indata[:, 0].copy())

    def worker():
        while True:
            item = utterances.get()
            if item is None:
                return
            audio, speech_end = item
            t0 = time.monotonic()
            try:
                text = recognizer.transcribe(audio)
            except Exception as e:
                log(f"❌ 인식 오류: {e}")
                continue
            stt = time.monotonic() - t0
            if cfg.reload():
                player.preload()
            if not text or is_ignored(text, cfg.settings.get("ignore_texts", [])):
                if text:
                    log(f"🙈 무시: “{text}”")
                continue
            handle_text(text, cfg, player, stt=stt, speech_end=speech_end)

    threading.Thread(target=worker, daemon=True).start()

    log(f"🎙️  입력 장치: {info['name']} ({sr}Hz)")
    out = s.get("output_device")
    log(f"🔈 출력 장치: {out if out is not None else '시스템 기본'}")
    log("말해 보세요. 종료는 Ctrl+C")

    pre = deque(maxlen=max(1, int(s["pre_roll"] / 0.03)))
    buf, speaking, silent_blocks = [], False, 0
    pending = np.zeros(0, dtype=np.float32)

    with sd.InputStream(device=in_dev, channels=1, samplerate=sr, dtype="float32",
                        blocksize=block, callback=callback):
        while True:
            chunk = frames.get()
            pending = np.concatenate([pending, chunk])
            while len(pending) >= block:
                b, pending = pending[:block], pending[block:]
                s = cfg.settings

                # 프로그램이 낸 소리를 다시 인식하지 않도록 재생 중에는 듣지 않음
                if s.get("ignore_while_playing") and player.is_playing():
                    pre.clear(); buf, speaking, silent_blocks = [], False, 0
                    continue

                db = 20 * np.log10(np.sqrt(np.mean(b ** 2)) + 1e-10)
                loud = db > s["start_threshold_db"]

                if not speaking:
                    pre.append(b)
                    if loud:
                        speaking, buf, silent_blocks = True, list(pre), 0
                    continue

                buf.append(b)
                silent_blocks = 0 if loud else silent_blocks + 1
                length = len(buf) * 0.03
                ended = silent_blocks * 0.03 >= s["silence_duration"]
                if ended or length >= s["max_speech"]:
                    speech_len = length - silent_blocks * 0.03
                    if speech_len >= s["min_speech"]:
                        # 실제로 말이 끝난 시점(무음 대기 시간 이전)부터 지연시간을 잰다
                        speech_end = time.monotonic() - silent_blocks * 0.03
                        utterances.put((to_16k(np.concatenate(buf), sr), speech_end))
                        log(f"… 인식 중 ({speech_len:.1f}초 분량)")
                    pre.clear(); buf, speaking, silent_blocks = [], False, 0


def handle_text(text, cfg, player, stt=None, speech_end=None):
    cmd, score, phrase = find_match(text, cfg.commands, cfg.settings["match_threshold"])
    suffix = f"  [인식 {stt:.2f}초]" if stt is not None else ""
    played = False
    if cmd:
        log(f"🗣️  “{text}” → ✅ {cmd['name']} (‘{phrase}’, {score:.0f}점){suffix}")
        played = player.play(cmd)
    elif cfg.fallback:
        log(f"🗣️  “{text}” → 매칭 없음 (최고 {score:.0f}점) → fallback{suffix}")
        played = player.play(cfg.fallback)
    else:
        log(f"🗣️  “{text}” → 매칭 없음 (최고 {score:.0f}점){suffix}")
    if played and speech_end is not None:
        log(f"⏱️  말 끝 → 재생 시작: {time.monotonic() - speech_end:.2f}초")


# ─────────────────────────────── 실행 ───────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="한국어 음성인식 → 음원 재생")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="설정 파일 경로 (기본: commands.json)")
    ap.add_argument("--list-devices", action="store_true", help="오디오 장치 목록 보기")
    ap.add_argument("--type", action="store_true", help="마이크 대신 키보드로 문장을 입력해 테스트")
    ap.add_argument("--test", metavar="문장", help="한 문장만 매칭 테스트")
    ap.add_argument("--no-play", action="store_true", help="매칭만 하고 소리는 재생하지 않음")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    cfg = Config(args.config)
    log(f"📄 설정: {cfg.path} (명령 {len(cfg.commands)}개)")
    player = Player(cfg, dry_run=args.no_play)
    if not args.no_play:
        player.preload()

    if args.test is not None:
        handle_text(args.test, cfg, player)
        if not args.no_play:
            time.sleep(max(0.0, player.busy_until - time.monotonic()))
        return

    if args.type:
        log("문장을 입력하고 Enter (종료: Ctrl+C 또는 빈 줄에서 Ctrl+D)")
        try:
            for line in sys.stdin:
                if cfg.reload() and not args.no_play:
                    player.preload()
                if line.strip():
                    handle_text(line.strip(), cfg, player)
        except KeyboardInterrupt:
            pass
        return

    recognizer = Recognizer(cfg)
    recognizer.warmup()
    try:
        run_mic(cfg, player, recognizer)
    except KeyboardInterrupt:
        log("종료합니다.")


if __name__ == "__main__":
    main()
