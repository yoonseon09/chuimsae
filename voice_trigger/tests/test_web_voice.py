"""Exercise the actual web protocol and deadline decisions without sockets/GPU/audio."""
import asyncio
import json
import queue
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from aiohttp import WSMsgType, web
from aiohttp.test_utils import make_mocked_request

from trigger_core import Command, Config, DEFAULTS
from web_backend import process_capture, web_config
from web_voice import Capture, ProtocolError, VoiceService, create_app, create_setup_app

HERE = Path(__file__).resolve().parents[1]


def pcm(seconds=.32, rate=16000, tail=0):
    signal = np.sin(np.arange(round(seconds * rate)) * 2 * np.pi * 200 / rate) * 10000
    return np.concatenate([signal, np.zeros(round(tail * rate))]).astype('<i2').tobytes()


class CaptureTests(unittest.TestCase):
    def test_stream_then_release_preserves_bytes_and_clock(self):
        clip = pcm(rate=48000)
        capture = Capture(dict(id='press', sample_rate=48000), 100)
        for start in range(0, len(clip), 2048):
            capture.append(clip[start:start + 2048], 100.3)
        job = capture.finish(dict(id='press', samples=len(clip) // 2,
                                  released_server_ms=100300, clock_uncertainty_ms=10), 100.4, 7)
        self.assertEqual(job['pcm'], clip)
        self.assertEqual(job['uid'], 7)
        self.assertAlmostEqual(job['released_at'], 100.29)
        self.assertTrue(job['timing_synced'])

    def test_sample_loss_and_invalid_pcm_are_rejected(self):
        capture = Capture(dict(id='press', sample_rate=16000), 100)
        with self.assertRaises(ProtocolError):
            capture.append(b'odd', 100.1)
        capture.append(pcm(), 100.32)
        with self.assertRaises(ProtocolError):
            capture.finish(dict(id='press', samples=1), 100.4, 1)

    def test_too_long_capture_is_rejected(self):
        capture = Capture(dict(id='press', sample_rate=16000), 100)
        with self.assertRaises(ProtocolError):
            capture.append(pcm(), 116)

    def test_empty_release_and_invalid_rates_are_rejected(self):
        for rate in (True, 1, 200000, float('nan'), '48000'):
            with self.assertRaises(ProtocolError):
                Capture(dict(id='press', sample_rate=rate), 100)
        capture = Capture(dict(id='press', sample_rate=16000), 100)
        with self.assertRaises(ProtocolError):
            capture.finish(dict(id='press', samples=0), 100, 1)

    def test_invalid_clock_does_not_extend_release_into_future(self):
        capture = Capture(dict(id='press', sample_rate=16000), 100)
        capture.append(pcm(), 100.4)
        job = capture.finish(dict(id='press', samples=len(capture.pcm) // 2,
                                  released_server_ms=float('nan'), clock_uncertainty_ms=0), 100.5, 1)
        self.assertFalse(job['timing_synced'])
        self.assertEqual(job['released_at'], 100.5)

    def test_web_settings_do_not_change_terminal_settings(self):
        original = Config.load(HERE / 'commands.json')
        adjusted = web_config(HERE / 'commands.json')
        self.assertEqual(original.settings['playback_gap'], 2)
        self.assertEqual(adjusted.settings['playback_gap'], 0)
        self.assertEqual(adjusted.settings['max_latency'], 3)
        self.assertEqual(original.settings['max_latency'], 10)


class BackendTests(unittest.TestCase):
    def run_capture(self, *, inference=0.2, voice='female', voice_delay=0, tail=0, text='얼쑤',
                    cancel_during=False, cancelled_before=False, silence=False, rate=16000):
        now = SimpleNamespace(value=100.)
        cancelled = threading.Event()
        if cancelled_before:
            cancelled.set()
        output, events = [], []
        cfg = SimpleNamespace(settings={**DEFAULTS, 'max_latency': 3, 'playback_gap': 0,
                                        'voice_mode': 'model'}, commands=(
            Command('얼쑤', ('얼쑤',), (Path('default.wav'),), voice_sounds={
                'male': (Path('male.wav'),), 'female': (Path('female.wav'),)}),))

        class Vad:
            def reset(self): pass
            def __call__(self, samples): return float(np.max(np.abs(samples)) > .01)

        class Voice:
            def classify(self, audio):
                now.value += voice_delay
                return {'label': voice, 'score': .95}

        class Player:
            def __init__(self):
                self.blocked_until = self.next_playback_at = -float('inf')
                self.notifications = queue.Queue()
                self.lock = threading.Lock()
                self.pending = None

            def schedule(self, item, command, deadline):
                if now.value >= deadline:
                    return None
                output.append(command.sounds[0].name)
                self.notifications.put(('sound_selected', {}))
                self.notifications.put(('playback_started', {'scheduled_dac_latency': now.value - item.speech_end}))
                self.blocked_until = now.value + .1
                self.next_playback_at = self.blocked_until
                return self.blocked_until

        transcriptions = []
        def transcribe(audio):
            transcriptions.append(audio)
            now.value += inference
            if cancel_during:
                cancelled.set()
            return dict(text=text, segments=[dict(text=text, avg_logprob=-.1,
                                                  no_speech_prob=.01, compression_ratio=1)])

        def sleep(seconds): now.value += seconds
        clip = bytes(len(pcm(rate=rate))) if silence else pcm(rate=rate, tail=tail)
        request = dict(pcm=clip, sample_rate=rate, released_at=100., uid=1,
                       timing_synced=True, clock_uncertainty_ms=5)
        runtime = dict(cfg=cfg, vad=Vad(), voice=Voice(), player=Player(), transcribe=transcribe)
        process_capture(request, runtime, events.append, cancelled, clock=lambda: now.value, sleep=sleep)
        return output, events, transcriptions

    def test_valid_press_routes_voice_to_correct_mac_sound(self):
        output, events, _ = self.run_capture()
        self.assertEqual(output, ['female.wav'])
        playback = next(e for e in events if e['type'] == 'playback')
        self.assertLess(playback['latency_ms'], 3000)

    def test_slow_transcription_cannot_play_late(self):
        output, events, _ = self.run_capture(inference=3.1)
        self.assertFalse(output)
        self.assertEqual(events[-1]['kind'], 'error')

    def test_voice_classification_time_counts_towards_budget(self):
        output, events, _ = self.run_capture(inference=2.7, voice_delay=.4)
        self.assertFalse(output)
        self.assertEqual(events[-1]['kind'], 'error')

    def test_trailing_silence_counts_from_speech_end(self):
        output, events, _ = self.run_capture(inference=1.1, tail=2.1)
        self.assertFalse(output)
        self.assertEqual(events[-1]['kind'], 'error')

    def test_cancelled_or_disconnected_request_does_not_play(self):
        for kwargs in ({'cancelled_before': True}, {'cancel_during': True}):
            output, _, _ = self.run_capture(**kwargs)
            self.assertFalse(output)

    def test_unknown_voice_does_not_fall_back_to_other_sound(self):
        output, events, _ = self.run_capture(voice='unknown')
        self.assertFalse(output)
        self.assertEqual(events[-1]['kind'], 'unmatched')

    def test_silence_and_unregistered_word_do_not_play(self):
        output, _, transcriptions = self.run_capture(silence=True)
        self.assertFalse(output)
        self.assertFalse(transcriptions)
        output, events, _ = self.run_capture(text='안녕하세요')
        self.assertFalse(output)
        self.assertEqual(events[-1]['kind'], 'unmatched')

    def test_phone_sample_rate_is_resampled_for_models(self):
        output, _, transcriptions = self.run_capture(rate=44100)
        self.assertEqual(output, ['female.wav'])
        self.assertAlmostEqual(len(transcriptions[0]) / 16000, .32, places=3)


class FakeSocket:
    def __init__(self, incoming):
        self.incoming = iter(incoming)
        self.sent = []
        self.closed = False
        self.close_code = None
    async def prepare(self, request): pass
    def __bool__(self): return False  # aiohttp responses are empty mutable mappings.
    async def receive_json(self): return next(self.incoming)
    async def send_json(self, data): self.sent.append(data)
    async def close(self, code=1000, message=b''):
        self.closed, self.close_code = True, code
    def __aiter__(self): return self
    async def __anext__(self):
        try: return next(self.incoming)
        except StopIteration: raise StopAsyncIteration


def message(data):
    return SimpleNamespace(type=WSMsgType.BINARY if isinstance(data, bytes) else WSMsgType.TEXT,
                           data=data if isinstance(data, bytes) else json.dumps(data))


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_watchdog_cancels_slow_inference_and_reports_failure(self):
        service = VoiceService(HERE / 'commands.json', 'secret')
        service.owner = FakeSocket([])
        cancelled = threading.Event()
        service.backend = SimpleNamespace(cancelled=cancelled,
                                          connection=SimpleNamespace(poll=lambda: False),
                                          process=SimpleNamespace(is_alive=lambda: True))
        service.state, service.phase = 'busy', 'processing'
        service.work_started = service.released_at = time.monotonic() - 3.1
        with patch('web_voice.asyncio.sleep', new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await service.monitor()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(service.timed_out)
        self.assertEqual(service.owner.sent[0]['kind'], 'error')

    async def test_disconnect_during_inference_cancels_late_output(self):
        service = VoiceService(HERE / 'commands.json', 'secret')
        owner = FakeSocket([])
        service.owner, service.phase = owner, 'processing'
        service.backend = SimpleNamespace(cancelled=threading.Event())
        await service.abandon(owner)
        self.assertTrue(service.backend.cancelled.is_set())
        self.assertIsNone(service.owner)

    async def test_playback_notifications_reach_an_empty_aiohttp_response(self):
        service = VoiceService(HERE / 'commands.json', 'secret', preview=True)
        service.owner = FakeSocket([])
        await service.tell_owner(dict(type='playback', command='얼쑤', latency_ms=450))
        self.assertEqual(service.owner.sent[0]['type'], 'playback')

    async def exchange(self, messages, *, first=None, service=None):
        service = service or VoiceService(HERE / 'commands.json', 'secret', preview=True)
        ws = FakeSocket([first if first is not None else dict(type='auth', key='secret'), *map(message, messages)])
        request = make_mocked_request('GET', '/ws', headers={'Host': 'localhost'})
        with patch('web_voice.web.WebSocketResponse', return_value=ws):
            await service.websocket(request)
        return service, ws

    async def test_preview_stream_is_received_without_fake_recognition(self):
        clip = pcm()
        service, ws = await self.exchange([dict(type='begin', id='press', sample_rate=16000),
                                          clip, dict(type='end', id='press', samples=len(clip) // 2)])
        self.assertEqual(service.state, 'ready')
        self.assertTrue(any(m['type'] == 'capture_ready' for m in ws.sent))
        self.assertTrue(any(m.get('kind') == 'preview' for m in ws.sent))
        self.assertFalse(any(m['type'] in ('recognized', 'playback') for m in ws.sent))

    async def test_missing_or_wrong_key_cannot_capture(self):
        for first in (dict(type='auth', key='wrong'), dict(type='auth', key='한글'), {}):
            _, ws = await self.exchange([], first=first)
            self.assertEqual(ws.close_code, 1008)

    async def test_cancel_and_disconnect_release_microphone_owner(self):
        for ending in ([dict(type='cancel', id='press')], []):
            service, _ = await self.exchange([dict(type='begin', id='press', sample_rate=16000), *ending])
            self.assertIsNone(service.owner)
            self.assertEqual(service.state, 'ready')

    async def test_malformed_end_discards_input_and_releases_owner(self):
        service, ws = await self.exchange([dict(type='begin', id='press', sample_rate=16000), pcm(),
                                          dict(type='end', id='press', samples=1)])
        self.assertTrue(any(m['type'] == 'error' for m in ws.sent))
        self.assertIsNone(service.capture)
        self.assertEqual(service.state, 'ready')

    async def test_binary_without_press_cannot_trigger_output(self):
        _, ws = await self.exchange([pcm()])
        self.assertTrue(any(m['type'] == 'error' for m in ws.sent))
        self.assertFalse(any(m['type'] == 'playback' for m in ws.sent))

    async def test_second_phone_cannot_take_over_active_press(self):
        service = VoiceService(HERE / 'commands.json', 'secret', preview=True)
        owner = object()
        service.owner, service.state, service.phase = owner, 'busy', 'capturing'
        service, ws = await self.exchange([dict(type='begin', id='intruder', sample_rate=16000)], service=service)
        self.assertIs(service.owner, owner)
        self.assertTrue(any(m['type'] == 'rejected' for m in ws.sent))

    async def test_other_origin_is_rejected(self):
        service = VoiceService(HERE / 'commands.json', 'secret', preview=True)
        request = make_mocked_request('GET', '/ws', headers={'Host': 'localhost', 'Origin': 'https://other.test'})
        with self.assertRaises(web.HTTPForbidden):
            await service.websocket(request)

    async def test_setup_routes_never_expose_private_key(self):
        service = VoiceService(HERE / 'commands.json', 'secret', preview=True)
        app = create_setup_app(service, HERE / '.web-local', 'localhost', 8765, 'test')
        for route in app.router.routes():
            self.assertNotIn('root.key', str(route.resource))
        app = create_app(service)
        self.assertFalse(any('static' in str(route.resource).lower() for route in app.router.routes()))

    async def test_setup_serves_public_profile_only_with_connection_key(self):
        service = VoiceService(HERE / 'commands.json', 'secret', preview=True)
        app = create_setup_app(service, HERE / '.web-local', 'localhost', 8765, 'test')
        for name, expected in (('/chuimsae.mobileconfig?key=secret', 200),
                               ('/chuimsae-root.cer?key=secret', 200),
                               ('/root.key?key=secret', 404),
                               ('/chuimsae.mobileconfig?key=wrong', 403)):
            request = make_mocked_request('GET', name)
            match = await app.router.resolve(request)
            request._match_info = match
            try:
                response = await match.handler(request)
                self.assertEqual(response.status, expected)
            except web.HTTPException as exc:
                self.assertEqual(exc.status, expected)


if __name__ == '__main__':
    unittest.main()
