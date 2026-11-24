"""One isolated Mac inference/output process for the phone push-to-talk UI."""
from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time
from dataclasses import replace

import numpy as np

from trigger_core import Config, FRAME, SR, Utterance, evaluate_commands, route_command

BUDGET = 3.0
MAX_CAPTURE = 10.5


def web_config(path):
    cfg = Config.load(path)
    return replace(cfg, settings={**cfg.settings, 'max_latency': BUDGET, 'playback_gap': 0.0})


def process_capture(request, runtime, emit, cancelled, clock=time.monotonic, sleep=time.sleep):
    """Never substitute an unclassified voice or play a result after its deadline.

    Playback remains ordered within a press; each new press is a separate request.
    The phone and Mac synchronize monotonic clocks before measuring the deadline.
    DAC timestamps are estimates, not acoustic measurements of an external speaker.
    """
    from voice_trigger import resample

    cfg, vad, player = runtime['cfg'], runtime['vad'], runtime['player']
    raw = np.frombuffer(request['pcm'], dtype='<i2').astype(np.float32) / 32768
    audio = resample(raw, request['sample_rate'], SR)
    vad.reset()
    last_voice = None
    voiced = 0
    for offset in range(0, len(audio), FRAME):
        if cancelled.is_set():
            return
        frame = audio[offset:offset + FRAME]
        if len(frame) < FRAME:
            frame = np.pad(frame, (0, FRAME - len(frame)))
        if vad(frame) >= cfg.settings['vad_end']:
            last_voice = min(len(audio), offset + FRAME)
            voiced += min(FRAME, len(audio) - offset)
    if last_voice is None or voiced / SR < cfg.settings['min_speech']:
        emit({'type': 'outcome', 'kind': 'empty', 'title': '목소리를 듣지 못했어요',
              'detail': '원을 누른 채 조금 더 가까이 말해주세요.'})
        return
    tail = (len(audio) - last_voice) / SR
    speech_end = request['released_at'] - tail
    deadline = speech_end + BUDGET - cfg.settings['output_margin']
    item = Utterance(audio, request['released_at'] - len(audio) / SR, speech_end,
                     voiced / SR, uid=request['uid'])

    def late():
        if clock() >= deadline:
            emit({'type': 'outcome', 'kind': 'error', 'title': '3초 안에 처리하지 못했어요',
                  'detail': '늦어진 음원은 재생하지 않았어요. Mac의 인식 속도를 확인해주세요.'})
            return True
        return False

    if late() or cancelled.is_set():
        return
    result = runtime['transcribe'](audio)
    if cancelled.is_set() or late():
        return
    commands, reason = evaluate_commands(result, item, cfg)
    if commands and runtime.get('voice'):
        result['voice'] = runtime['voice'].classify(audio)
    if cancelled.is_set() or late():
        return
    routed = []
    for command in commands:
        candidate, reason = route_command(command, result.get('voice'), cfg.settings)
        if candidate:
            routed.append(candidate)
    if not routed:
        reasons = {
            'not_registered_interjection': '‘얼쑤’, ‘좋다’, ‘잘한다’ 중 하나를 말해주세요.',
            'voice_unknown': '목소리를 구분하지 못했어요. 조금 더 또렷하게 말해주세요.',
            'voice_sound_missing': '해당 목소리에 맞는 음원 파일이 없어요.',
        }
        emit({'type': 'outcome', 'kind': 'unmatched', 'title': f'“{result.get("text", "") or "…"}”',
              'detail': reasons.get(reason, '인식이 확실하지 않아 재생하지 않았어요.')})
        return
    emit({'type': 'recognized', 'text': result['text'], 'detail': 'Mac 스피커로 보내고 있어요'})
    played = 0
    for command in routed:
        if cancelled.is_set():
            return
        while clock() < max(player.blocked_until, player.next_playback_at):
            if cancelled.is_set() or late():
                return
            sleep(0.01)
        if late():
            return
        blocked = player.schedule(item, command, deadline=deadline)
        if blocked is None:
            emit({'type': 'outcome', 'kind': 'error', 'title': '출력을 시작하지 못했어요',
                  'detail': 'Mac의 스피커와 출력 장치 설정을 확인해주세요.'})
            return
        started = False
        while not started:
            if cancelled.is_set():
                # Cancel a still-pending callback; never start a disconnected request.
                with player.lock:
                    player.pending = None
                return
            try:
                event, fields = player.notifications.get(timeout=0.02)
            except queue.Empty:
                if late():
                    with player.lock:
                        player.pending = None
                    return
                continue
            if event == 'playback_started':
                started = True
                played += 1
                emit({'type': 'playback', 'command': command.name,
                      'latency_ms': fields['scheduled_dac_latency'] * 1000,
                      'timing_synced': request['timing_synced'],
                      'clock_uncertainty_ms': request['clock_uncertainty_ms']})
            elif event in ('deadline_at_output', 'output_clock_error'):
                emit({'type': 'outcome', 'kind': 'error', 'title': '출력 기한을 지키지 못했어요',
                      'detail': '늦어진 음원은 재생하지 않았어요.'})
                return
        # Keep the next capture closed during playback, to prevent speaker feedback.
        while clock() < player.blocked_until:
            if cancelled.is_set():
                return
            sleep(0.01)
    emit({'type': 'outcome', 'kind': 'played', 'title': '재생 완료', 'detail': '', 'count': played})


def backend_worker(connection, config_path, cancelled):
    player = None
    try:
        from voice_trigger import Player, Silero, local_model
        from voice_profile import VoiceClassifier
        import mlx_whisper

        cfg = web_config(config_path)
        vad = Silero()
        voice = VoiceClassifier(cfg.settings) if cfg.settings['voice_mode'] == 'model' else None
        kwargs = dict(path_or_hf_repo=local_model(cfg.settings['model']), language='ko',
                      temperature=0.0, condition_on_previous_text=False, initial_prompt=None,
                      word_timestamps=cfg.settings['word_timestamps'], verbose=None,
                      compression_ratio_threshold=cfg.settings['max_compression_ratio'],
                      logprob_threshold=cfg.settings['min_avg_logprob'],
                      no_speech_threshold=cfg.settings['max_no_speech_prob'])

        def transcribe(audio):
            return mlx_whisper.transcribe(audio, **kwargs)

        transcribe(np.zeros(SR, np.float32))
        player = Player(cfg)
        runtime = dict(cfg=cfg, vad=vad, voice=voice, player=player, transcribe=transcribe)
        connection.send({'type': 'backend_ready'})
        while True:
            request = connection.recv()
            if request is None:
                return
            try:
                process_capture(request, runtime, connection.send, cancelled)
            except Exception as exc:
                connection.send({'type': 'outcome', 'kind': 'error', 'title': '인식 중 문제가 생겼어요',
                                 'detail': str(exc).splitlines()[0][:300]})
            # A cancelled request may already be audible. Keep input closed until quiet.
            while time.monotonic() < player.blocked_until:
                time.sleep(0.01)
            connection.send({'type': 'backend_ready'})
    except BaseException as exc:
        try:
            connection.send({'type': 'backend_error', 'detail': str(exc).splitlines()[0][:400]})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if player:
            player.close()
        connection.close()


class Backend:
    def __init__(self, config_path):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.cancelled = context.Event()
        self.process = context.Process(target=backend_worker, args=(child, str(config_path), self.cancelled))
        self.process.start()
        child.close()

    def close(self):
        self.cancelled.set()
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        self.connection.close()
