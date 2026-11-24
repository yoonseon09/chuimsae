#!/usr/bin/env python3
"""Silero VAD → MLX Whisper → 문장 내 추임새 판정 → 중복 병합·순차 음원 출력."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import queue
import random
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from trigger_core import (Config, FRAME, SR, STEP, PlaybackQueue, Segmenter,
                          evaluate_commands, resolve_path, route_command, text_commands)
from voice_profile import VoiceClassifier

HERE = Path(__file__).resolve().parent


def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)


class Events:
    def __init__(self, cfg):
        self.counts = Counter()
        self.handle = None
        name = cfg.settings['event_log']
        if name:
            path = resolve_path(cfg.path.parent, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open('a', encoding='utf-8', buffering=1)
        self.emit('session', model=cfg.settings['model'], settings=cfg.settings)

    def emit(self, event, **values):
        self.counts[event] += 1
        if self.handle:
            self.handle.write(json.dumps({'event': event, 'time': time.time(),
                                          'monotonic': time.monotonic(), **values},
                                         ensure_ascii=False, allow_nan=False) + '\n')

    def close(self):
        log('처리 통계: ' + json.dumps(dict(self.counts), ensure_ascii=False))
        if self.handle:
            self.handle.close()


def resample(audio, original_sr, target_sr):
    if original_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly
    divisor = math.gcd(original_sr, target_sr)
    return resample_poly(audio, target_sr // divisor, original_sr // divisor,
                         axis=0).astype(np.float32)


class Silero:
    def __init__(self):
        # 패키지의 검증된 JIT 파일만 로드한다. torch.hub의 원격 Python 실행을 사용하지 않는다.
        try:
            dist = metadata.distribution('silero-vad')
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError('Silero VAD가 없습니다. 설치.command를 실행하세요.') from exc
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        path = dist.locate_file('silero_vad/data/silero_vad.jit')
        if not Path(path).is_file():
            raise RuntimeError(f'Silero 모델 파일이 없습니다: {path}')
        self.model = torch.jit.load(str(path), map_location='cpu').eval()
        self.reset()

    def reset(self):
        self.model.reset_states()

    def __call__(self, audio):
        with self.torch.inference_mode():
            return float(self.model(self.torch.from_numpy(audio.copy()), SR).item())


def local_model(model):
    """이미 받은 모델을 우선 사용해 실행 때 불필요한 Hub 통신을 피한다."""
    if Path(model).is_dir():
        return str(Path(model).resolve())
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:
        return model  # 첫 실행 때만 mlx-whisper가 공식 Hub에서 받는다.


def recognition_worker(connection, settings):
    try:
        voice_classifier = VoiceClassifier(settings) if settings['voice_mode'] == 'model' else None
        import mlx_whisper
        model = local_model(settings['model'])
        kwargs = dict(path_or_hf_repo=model, language='ko', temperature=0.0,
                      condition_on_previous_text=False, initial_prompt=None,
                      word_timestamps=settings['word_timestamps'], verbose=None,
                      compression_ratio_threshold=settings['max_compression_ratio'],
                      logprob_threshold=settings['min_avg_logprob'],
                      no_speech_threshold=settings['max_no_speech_prob'])
        mlx_whisper.transcribe(np.zeros(SR, np.float32), **kwargs)
        connection.send(('ready', model))
        while True:
            item = connection.recv()
            if item is None:
                break
            started = time.monotonic()
            try:
                result = mlx_whisper.transcribe(item.audio, **kwargs)
                # MLX array를 넘기지 않는다. 프로세스 간 결과는 작은 Python 객체만 전송한다.
                segments = []
                for segment in result.get('segments', []):
                    segments.append({k: segment.get(k) for k in
                                     ('text', 'avg_logprob', 'no_speech_prob', 'compression_ratio', 'words')})
                response = {'text': result.get('text', '').strip(), 'segments': segments}
                if voice_classifier:
                    response['voice'] = voice_classifier.classify(item.audio)
                connection.send(('result', item.uid, response, time.monotonic() - started))
            except Exception as exc:
                connection.send(('error', item.uid, str(exc)))
    except BaseException as exc:
        try:
            connection.send(('fatal', str(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class Recognizer:
    """동시에 한 작업만 전송한다. 부모 프로세스가 대기 발화를 순서대로 보관한다."""
    def __init__(self, settings):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=recognition_worker, args=(child, settings), daemon=True)
        self.process.start()
        child.close()
        self.current = None
        self.sent_at = None
        try:
            limit = time.monotonic() + 600
            log('음성 인식·목소리 분류 모델 준비 중… 처음에는 다운로드 시간이 필요할 수 있습니다.')
            while not self.connection.poll(0.1):
                self.check_alive()
                if time.monotonic() > limit:
                    raise RuntimeError('모델 준비가 600초를 넘었습니다. 모델 다운로드·메모리를 확인하세요.')
            message = self.connection.recv()
            if message[0] != 'ready':
                raise RuntimeError(f'음성 모델 준비 실패: {message}')
            log('음성 모델 준비 완료')
        except BaseException:
            self.close()
            raise

    def check_alive(self):
        if not self.process.is_alive():
            raise RuntimeError(f'음성 인식 프로세스가 종료되었습니다 (exit={self.process.exitcode}). '
                               'Metal 사용 가능 여부와 설치 환경을 --doctor로 확인하세요.')

    def send(self, item):
        if self.current is not None:
            raise RuntimeError('인식 중에는 추가 작업을 전송할 수 없습니다')
        self.check_alive()
        self.current = item
        self.sent_at = time.monotonic()
        self.connection.send(item)

    def poll(self):
        self.check_alive()
        if not self.connection.poll():
            if self.current is not None and time.monotonic() - self.sent_at > 12:
                raise RuntimeError('인식 작업이 12초를 넘었습니다. 준비된 모델의 속도를 확인하세요.')
            return None
        message = self.connection.recv()
        item, self.current = self.current, None
        if message[0] == 'fatal':
            raise RuntimeError(str(message[1]))
        if item is None or message[1] != item.uid:
            raise RuntimeError('음성 인식 결과 ID가 일치하지 않습니다')
        return item, message

    def transcribe(self, item):
        self.send(item)
        while True:
            response = self.poll()
            if response:
                return response[1]
            time.sleep(0.01)

    def close(self):
        if self.process.is_alive():
            # 진행 중 추론·대기 상태 모두 확실히 종료한다. GPU 작업 완료를 무기한 기다리지 않는다.
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        self.connection.close()


def resolve_device(spec, kind):
    import sounddevice as sd
    if spec is None:
        device = sd.default.device[0 if kind == 'input' else 1]
        if device < 0:
            raise RuntimeError(f'{kind} 기본 장치가 없습니다. macOS 오디오 장치·마이크 권한을 확인하세요.')
        return device
    if isinstance(spec, int):
        return spec
    matches = [i for i, d in enumerate(sd.query_devices())
               if d[f'max_{kind}_channels'] > 0 and spec.casefold() in d['name'].casefold()]
    if len(matches) != 1:
        raise RuntimeError(f'{kind} 장치 이름이 없거나 여러 개와 일치합니다: {spec!r}. --list-devices로 확인하세요.')
    return matches[0]


class Player:
    """영구 출력 스트림. 첫 샘플을 보내기 직전에도 DAC 시각 기준으로 기한을 검사한다."""
    def __init__(self, cfg, dry=False):
        self.cfg, self.dry = cfg, dry
        self.lock = threading.Lock()
        self.notifications = queue.Queue(maxsize=32)
        self.pending = None
        self.position = 0
        self.blocked_until = -math.inf
        self.next_playback_at = -math.inf
        self.stream = None
        self.cache = {}
        if dry:
            self.sr, self.channels = 48000, 2
        else:
            import sounddevice as sd
            device = resolve_device(cfg.settings['output_device'], 'output')
            info = sd.query_devices(device)
            self.sr = int(info['default_samplerate'])
            self.channels = min(2, info['max_output_channels'])
            if self.channels < 1:
                raise RuntimeError('선택한 장치가 오디오 출력을 지원하지 않습니다')
            self.stream = sd.OutputStream(device=device, samplerate=self.sr,
                                          channels=self.channels, dtype='float32', blocksize=256,
                                          latency='low', callback=self.callback)
        try:
            self.preload()
            if self.stream:
                self.stream.start()
        except BaseException:
            self.close()
            raise

    def preload(self):
        import soundfile as sf
        for cmd in self.cfg.commands:
            for path in cmd.all_sounds:
                data, sr = sf.read(path, dtype='float32', always_2d=True)
                if not len(data) or not np.isfinite(data).all():
                    raise ValueError(f'비어 있거나 손상된 음원: {path}')
                if data.shape[1] != self.channels:
                    data = np.repeat(data.mean(axis=1, keepdims=True), self.channels, axis=1)
                data = resample(data, sr, self.sr)
                active = np.flatnonzero(np.max(np.abs(data), axis=1) > 10 ** (-45 / 20))
                if not len(active):
                    raise ValueError(f'소리가 없는 음원: {path}')
                start = max(0, active[0] - round(self.sr * 0.005))
                end = min(len(data), active[-1] + round(self.sr * 0.04) + 1)
                self.cache[(cmd.name, path)] = np.ascontiguousarray(np.clip(data[start:end] * cmd.volume, -1, 1))
        log(f'음원 {len(self.cache)}개 준비 완료')

    def notify(self, event, **fields):
        try:
            self.notifications.put_nowait((event, fields))
        except queue.Full:
            pass

    def schedule(self, item, cmd, deadline=None):
        now = time.monotonic()
        with self.lock:
            if self.pending is not None or now < max(self.blocked_until, self.next_playback_at):
                return None
            path = random.choice(cmd.sounds)
            data = self.cache[(cmd.name, path)]
            if deadline is None:
                deadline = item.speech_end + self.cfg.settings['max_latency'] - self.cfg.settings['output_margin']
            if now >= deadline:
                return None
            self.notify('sound_selected', uid=item.uid, command=cmd.name, sound=path.name)
            if self.dry:
                self.blocked_until = now + len(data) / self.sr + self.cfg.settings['echo_guard']
                self.notify('would_play', uid=item.uid, command=cmd.name,
                            decision_latency=now - item.speech_end)
            else:
                self.pending = (data, item, cmd, deadline)
                self.position = 0
                # 최대 예상 출력 지연까지 예약한다. 실제 첫 callback에서 DAC 기준으로 갱신한다.
                self.blocked_until = now + max(0.1, float(self.stream.latency)) + len(data) / self.sr + self.cfg.settings['echo_guard']
            self.next_playback_at = (self.blocked_until - self.cfg.settings['echo_guard']
                                     + self.cfg.settings['playback_gap'])
            return self.blocked_until

    def callback(self, outdata, frames, timing, status):
        outdata.fill(0)
        with self.lock:
            if self.pending is None:
                return
            data, item, cmd, deadline = self.pending
            if status:
                self.notify('output_status', detail=str(status))
            if self.position == 0:
                now = time.monotonic()
                lead = float(timing.outputBufferDacTime - timing.currentTime)
                if not math.isfinite(lead) or not 0 <= lead <= 1:
                    self.pending = None
                    self.notify('output_clock_error', uid=item.uid)
                    return
                dac_start = now + lead
                if dac_start >= deadline:
                    self.pending = None
                    self.notify('deadline_at_output', uid=item.uid)
                    return
                self.blocked_until = dac_start + len(data) / self.sr + self.cfg.settings['echo_guard']
                self.next_playback_at = dac_start + len(data) / self.sr + self.cfg.settings['playback_gap']
                self.notify('playback_started', uid=item.uid, command=cmd.name,
                            scheduled_dac_latency=dac_start - item.speech_end)
            count = min(frames, len(data) - self.position)
            outdata[:count] = data[self.position:self.position + count]
            self.position += count
            if self.position == len(data):
                self.pending = None

    def close(self):
        if self.stream:
            self.stream.abort()
            self.stream.close()
            self.stream = None


def drain_notifications(player, events):
    while True:
        try:
            name, fields = player.notifications.get_nowait()
        except queue.Empty:
            return
        events.emit(name, **fields)
        if name == 'playback_started':
            log(f"재생: {fields['command']} | 추정 DAC 출력 지연 {fields['scheduled_dac_latency']:.3f}초")
        elif name == 'would_play':
            log(f"재생 생략: {fields['command']} | 판정 지연 {fields['decision_latency']:.3f}초")
        elif name == 'deadline_at_output':
            log(f'재생 안 함: 출력 예정 시각이 {player.cfg.settings["max_latency"]:g}초 제한을 넘었습니다.')
        elif name == 'sound_selected':
            log(f"선택 음원: {fields['sound']}")
        else:
            log(f'출력 이벤트: {name} {fields}')


def record_results(item, message, cfg, events):
    if message[0] == 'error':
        events.emit('recognition_error', uid=item.uid, detail=message[2])
        log(f'인식 오류: {message[2]}')
        return []
    _, _, result, elapsed = message
    commands, reason = evaluate_commands(result, item, cfg)
    voice = result.get('voice')
    routed = []
    for command in commands:
        candidate, route_reason = route_command(command, voice, cfg.settings)
        events.emit('command_candidate', uid=item.uid, command=command.name, reason=route_reason)
        if candidate:
            routed.append(candidate)
        elif not routed:
            reason = route_reason
    if routed:
        reason = 'accepted'
    # 원시 음성은 저장하지 않는다. 진단 로그에는 인식 텍스트가 포함된다.
    events.emit('recognized', uid=item.uid, text=result['text'], reason=reason,
                inference_seconds=elapsed, start=item.start, speech_end=item.speech_end,
                command=routed[0].name if routed else None,
                commands=[c.name for c in routed], voice=voice,
                sound_candidates=[str(p) for c in routed for p in c.sounds])
    detail = ''
    if isinstance(voice, dict):
        label = {'male': '남성 추정', 'female': '여성 추정', 'unknown': '구분 불가'}.get(voice.get('label'), '구분 불가')
        score = voice.get('score')
        detail = f" | 목소리: {label}, 분류 점수 {score:.3f}" if isinstance(score, (int, float)) else f' | 목소리: {label}'
    log(f"인식: {result['text']!r} → {('추임새 확인: ' + ', '.join(c.name for c in routed)) if routed else reason} ({elapsed:.2f}초){detail}")
    return routed


def record_result(item, message, cfg, events):
    return next(iter(record_results(item, message, cfg, events)), None)


def run_mic(cfg, events, recognizer, vad, dry=False):
    import sounddevice as sd
    playback, segmenter = PlaybackQueue(cfg.settings), Segmenter(cfg.settings)
    discard_before = -math.inf
    last_voice_end = -math.inf
    device = resolve_device(cfg.settings['input_device'], 'input')
    sr = None
    for candidate in (SR, 48000, 32000):
        try:
            sd.check_input_settings(device=device, channels=1, dtype='float32', samplerate=candidate)
            sr = candidate
            break
        except sd.PortAudioError:
            continue
    if sr is None:
        raise RuntimeError('마이크가 16/32/48 kHz 입력을 지원하지 않습니다. 다른 입력 장치를 선택하세요.')
    block = int(sr * STEP)
    frames = queue.Queue(maxsize=8)
    overflow = threading.Event()

    def callback(indata, count, timing, status):
        now = time.monotonic()
        age = float(timing.currentTime - timing.inputBufferAdcTime)
        if status or not math.isfinite(age) or not 0 <= age <= 2:
            overflow.set()
            return
        captured_end = min(now, now - age + count / sr)
        try:
            frames.put_nowait((indata[:, 0].copy(), captured_end))
        except queue.Full:
            overflow.set()

    player = Player(cfg, dry)
    pending = deque()
    last_capture = None
    try:
        with sd.InputStream(device=device, channels=1, samplerate=sr, blocksize=block,
                            dtype='float32', callback=callback):
            log(f"준비 완료: {sd.query_devices(device)['name']} | 활성 단어: {', '.join(c.name for c in cfg.commands)}")
            log(f"정책: 문장 속 추임새 인식 | 같은 추임새 {cfg.settings['duplicate_window']:g}초 내 병합"
                f" | 앞 음원 종료 후 {cfg.settings['playback_gap']:g}초 간격")
            log('Ctrl+C로 종료합니다. 설정을 변경하면 프로그램을 다시 실행하세요.')
            report_voice_routes(cfg)
            while True:
                drain_notifications(player, events)
                discard_before = max(discard_before, player.blocked_until)
                for expired in playback.expire(time.monotonic()):
                    events.emit('playback_queue_expired', uid=expired.item.uid, command=expired.command.name)
                    log(f'재생 대기 만료: {expired.command.name}')
                response = recognizer.poll()
                if response:
                    item, message = response
                    commands = record_results(item, message, cfg, events)
                    for command in commands:
                        reason = ('playback_overlap' if item.start < discard_before
                                  else playback.offer(item, command, time.monotonic()))
                        if reason:
                            age = time.monotonic() - item.speech_end
                            events.emit(reason, uid=item.uid, command=command.name, elapsed_since_speech_end=age)
                            if reason == 'deadline':
                                log(f'재생 안 함: 말 끝부터 {age:.2f}초가 지나 {cfg.settings["max_latency"]:g}초 제한을 넘었습니다.')
                            else:
                                log(f'재생 안 함: {command.name} / {reason}')
                        else:
                            events.emit('queued', uid=item.uid, command=command.name, depth=len(playback.pending))
                            log(f'재생 대기: {command.name} ({len(playback.pending)}개)')
                if pending and recognizer.current is None:
                    waiting = pending.popleft()
                    if waiting.start < discard_before:
                        events.emit('playback_overlap', uid=waiting.uid)
                    elif time.monotonic() >= waiting.speech_end + cfg.settings['max_latency'] - cfg.settings['output_margin']:
                        events.emit('deadline', uid=waiting.uid)
                        log('인식 대기 중 기한이 지나 해당 발화를 건너뜁니다.')
                    else:
                        recognizer.send(waiting)
                now = time.monotonic()
                # 연속 발화를 먼저 수집·인식한 뒤 재생을 시작한다. 출력 간격 중에도 입력을 받는다.
                if (playback.pending and not pending and recognizer.current is None
                        and not segmenter.active and now - last_voice_end >= cfg.settings['burst_silence']):
                    request = playback.pending[0]
                    blocked = player.schedule(request.item, request.command, deadline=request.deadline)
                    if blocked is not None:
                        playback.pending.popleft()
                        discard_before = max(discard_before, blocked)
                        segmenter.reset()
                        vad.reset()
                if overflow.is_set():
                    overflow.clear()
                    # 캡처 유실이 있으면 끊어진 오디오를 이어 붙이지 않는다.
                    while True:
                        try:
                            frames.get_nowait()
                        except queue.Empty:
                            break
                    segmenter.reset()
                    vad.reset()
                    pending.clear()
                    discard_before = max(discard_before, time.monotonic())
                    events.emit('input_overflow')
                    continue
                try:
                    data, end = frames.get(timeout=0.01)
                except queue.Empty:
                    continue
                if end - STEP < discard_before:
                    continue
                if time.monotonic() - end > cfg.settings['max_frame_age']:
                    segmenter.reset()
                    vad.reset()
                    events.emit('stale_frame')
                    continue
                if last_capture is not None and abs(end - last_capture - STEP) > STEP / 2:
                    segmenter.reset()
                    vad.reset()
                    events.emit('capture_gap')
                last_capture = end
                audio = resample(data, sr, SR)
                if len(audio) != FRAME:
                    raise RuntimeError('입력 프레임 길이 오류')
                probability = vad(audio)
                if probability >= cfg.settings['vad_end']:
                    last_voice_end = end
                item = segmenter.feed(audio, end, probability)
                if item:
                    if len(pending) >= cfg.settings['queue_capacity']:
                        events.emit('recognition_queue_full', uid=item.uid)
                    else:
                        pending.append(item)
    finally:
        player.close()
        drain_notifications(player, events)


def analyze_file(path, cfg, events, recognizer, vad):
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError('파일이 비어 있거나 손상되어 있습니다')
    audio = resample(data.mean(axis=1), sr, SR)
    segmenter = Segmenter(cfg.settings)
    log('파일 분석 모드: 음원은 재생하지 않으며, 실제 마이크 응답 지연 측정은 아닙니다.')
    for offset in range(0, len(audio), FRAME):
        block = audio[offset:offset + FRAME]
        block = np.pad(block, (0, FRAME - len(block)))
        item = segmenter.feed(block, (offset + FRAME) / SR, vad(block))
        if item:
            record_results(item, recognizer.transcribe(item), cfg, events)
    item = segmenter.flush()
    if item:
        record_results(item, recognizer.transcribe(item), cfg, events)


def doctor(cfg):
    missing = []
    packages = ['mlx-whisper', 'mlx', 'sounddevice', 'soundfile', 'numpy', 'scipy', 'torch', 'silero-vad']
    if cfg.settings['voice_mode'] == 'model':
        packages += ['transformers', 'safetensors']
    for name in packages:
        try:
            log(f'{name}: {metadata.version(name)}')
        except metadata.PackageNotFoundError:
            missing.append(name)
            log(f'{name}: 설치 필요')
    log(f'Python: {sys.executable}')
    import sounddevice as sd
    devices = sd.query_devices()
    log(f'오디오 장치 {len(devices)}개')
    if not devices:
        log('현재 실행 환경에서 오디오 장치에 접근할 수 없습니다. macOS 터미널에서 실행하고 마이크 권한을 허용하세요.')
    log('설정 검증 완료: ' + ', '.join(c.name for c in cfg.commands))
    report_voice_routes(cfg)
    if missing:
        log('설치.command를 실행하세요. 누락: ' + ', '.join(missing))
    return 1 if missing or not devices else 0


def report_voice_routes(cfg):
    if cfg.settings['voice_mode'] == 'off':
        return
    log('학습 모델로 목소리 추정 켜짐. 구분 불가/음원 없음은 재생하지 않습니다.')
    for command in cfg.commands:
        for label, title in [('male', '남성 추정'), ('female', '여성 추정')]:
            sounds = command.voice_sounds.get(label, ())
            names = ', '.join(p.name for p in sounds) if sounds else '음원 없음 → 재생 안 함'
            log(f'{command.name} / {title}: {names}')


def inspect_voice(path, cfg):
    """Whisper/GPU/마이크 없이 녹음의 목소리 추정만 확인한다."""
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError('파일이 비어 있거나 손상되어 있습니다')
    audio = resample(data.mean(axis=1), sr, SR)
    if len(audio) > 30 * SR:
        raise ValueError('목소리 검사는 한 사람이 말하는 30초 이하 녹음을 사용하세요')
    log('목소리 분류 모델 준비 중… 최초에는 모델 다운로드가 필요합니다.')
    print(json.dumps(VoiceClassifier(cfg.settings).classify(audio), ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE / 'commands.json'))
    parser.add_argument('--list-devices', action='store_true')
    parser.add_argument('--doctor', action='store_true', help='설치·설정·장치 확인 (추론 검증은 --prepare)')
    parser.add_argument('--prepare', action='store_true', help='VAD·Whisper 모델 준비 후 종료')
    parser.add_argument('--no-play', action='store_true')
    parser.add_argument('--voice', choices=('male', 'female', 'unknown'),
                        help='--test/--type에서만 재생할 목소리를 수동 지정. 실제 목소리 검사 아님')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--test', metavar='문장', help='모델 없이 텍스트 판정만 확인. --no-play로 무음 검사')
    modes.add_argument('--type', action='store_true', help='모델 없이 텍스트를 입력하여 확인')
    modes.add_argument('--file', type=Path, help='녹음 파일 분석 (항상 음원 재생 안 함)')
    modes.add_argument('--inspect-voice', type=Path, help='Whisper 없이 단일 화자 녹음의 목소리 추정만 확인')
    args = parser.parse_args()
    if args.voice is not None and args.test is None and not args.type:
        parser.error('--voice는 --test 또는 --type에서만 사용할 수 있습니다')
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    cfg = Config.load(args.config)
    if args.inspect_voice:
        inspect_voice(args.inspect_voice, cfg)
        return 0
    if args.doctor:
        return doctor(cfg)
    if args.test is not None or args.type:
        log('텍스트 매칭 검사: 음성 인식 품질·실제 지연은 검사하지 않습니다.')
        player = Player(cfg, args.no_play)
        playback = PlaybackQueue(cfg.settings)
        try:
            lines = [args.test] if args.test is not None else sys.stdin
            for uid, line in enumerate(lines, 1):
                commands = text_commands(line.strip(), cfg.commands)
                log(f'{line.strip()!r} → {", ".join(c.name for c in commands) or "일치 없음"}')
                from trigger_core import Utterance
                now = time.monotonic()
                item = Utterance(np.empty(0), now, now, 0, uid=uid)
                for command in commands:
                    command, reason = route_command(command, {'label': args.voice}, cfg.settings)
                    if command is None:
                        log(f'재생 안 함: {reason}. 텍스트 검사는 --voice male/female을 지정하세요.')
                        continue
                    reason = playback.offer(item, command, now)
                    if reason:
                        log(f'재생 안 함: {command.name} / {reason}')
                while playback.pending:
                    now = time.monotonic()
                    playback.expire(now)
                    if not playback.pending:
                        break
                    request = playback.pending[0]
                    if player.schedule(request.item, request.command, deadline=request.deadline) is not None:
                        playback.pending.popleft()
                    else:
                        time.sleep(0.02)
                while time.monotonic() < player.blocked_until:
                    time.sleep(0.02)
                while not player.notifications.empty():
                    log(str(player.notifications.get_nowait()))
        finally:
            player.close()
        return 0
    # 누락된 VAD를 모델 다운로드·GPU 준비 전에 확인한다.
    vad = Silero()
    recognizer = Recognizer(cfg.settings)
    events = None
    try:
        if args.prepare:
            log('VAD·Whisper·설정된 목소리 모델 준비 완료. 마이크 실행이 가능합니다.')
            return 0
        events = Events(cfg)
        if args.file:
            analyze_file(args.file, cfg, events, recognizer, vad)
        else:
            run_mic(cfg, events, recognizer, vad, args.no_play)
    finally:
        recognizer.close()
        if events:
            events.close()
    return 0


if __name__ == '__main__':
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log('종료했습니다.')
    except Exception as exc:
        # 장치 오류(PortAudioError 등)도 성공으로 오인되지 않게 종료 코드 1로 보고한다.
        log(f'실행 실패 ({type(exc).__name__}): {exc}')
        raise SystemExit(1)
