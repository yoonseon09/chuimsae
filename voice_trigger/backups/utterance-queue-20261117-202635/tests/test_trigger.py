import json
import math
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from trigger_core import (Command, Config, DEFAULTS, FRAME, SR, STEP, PlaybackGate,
                          Segmenter, Utterance, evaluate, normalize, resolve_path, text_command)
from voice_trigger import Player, recognition_worker, resample


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.command = Command('얼쑤', ('얼쑤', '얼수'), (Path('unused.wav'),))
        self.cfg = SimpleNamespace(settings=dict(DEFAULTS), commands=(self.command,))
        self.cfg.settings['word_timestamps'] = True
        self.item = Utterance(np.zeros(SR, np.float32), 10, 11, 0.5, uid=1)
        self.result = {'text': '얼쑤!', 'segments': [{
            'text': '얼쑤!', 'avg_logprob': -0.2, 'no_speech_prob': 0.05,
            'compression_ratio': 1.0,
            'words': [{'word': '얼쑤!', 'probability': 0.8}]}]}

    def test_accept_only_registered_independent_phrase(self):
        self.assertEqual(text_command(' 얼쑤! ', self.cfg.commands), self.command)
        for text in ['얼쑤라고 말해 봐', '얼쑤 좋다', '어서 오세요', '얼쑥', '좋다', '얼쑤a', '얼쑤1', '', '얼쑤💬']:
            with self.subTest(text=text):
                self.assertIsNone(text_command(text, self.cfg.commands))

    def test_unicode(self):
        import unicodedata
        self.assertEqual(normalize(unicodedata.normalize('NFD', '얼쑤!')), '얼쑤')
        with tempfile.TemporaryDirectory() as folder:
            name = unicodedata.normalize('NFD', '남_얼쑤.wav')
            path = Path(folder, name)
            path.touch()
            self.assertTrue(resolve_path(Path(folder), '남_얼쑤.wav').samefile(path))

    def test_quality_accept(self):
        self.assertEqual(evaluate(self.result, self.item, self.cfg), (self.command, 'accepted'))

    def test_quality_reject_missing_or_low_metrics(self):
        for key, value, reason in [('avg_logprob', -1.5, 'low_logprob'),
                                   ('no_speech_prob', 0.9, 'no_speech'),
                                   ('compression_ratio', 4, 'repetition'),
                                   ('avg_logprob', float('nan'), 'missing_quality_metadata')]:
            result = json.loads(json.dumps(self.result))
            result['segments'][0][key] = value
            self.assertEqual(evaluate(result, self.item, self.cfg)[1], reason)
        self.assertEqual(evaluate({'text': '얼쑤'}, self.item, self.cfg)[1], 'missing_quality_metadata')
        self.result['segments'][0]['words'][0]['probability'] = 0.1
        self.assertEqual(evaluate(self.result, self.item, self.cfg)[1], 'low_word_probability')

    def test_fast_mode_requires_segment_quality_without_word_alignment(self):
        self.cfg.settings['word_timestamps'] = False
        self.result['segments'][0].pop('words')
        self.assertEqual(evaluate(self.result, self.item, self.cfg)[1], 'accepted')
        self.result['segments'][0]['avg_logprob'] = -1.5
        self.assertEqual(evaluate(self.result, self.item, self.cfg)[1], 'low_logprob')

    def test_long_sentence_fragments_never_play(self):
        segmenter = Segmenter(DEFAULTS)
        outputs = []
        for i in range(210):
            item = segmenter.feed(np.zeros(FRAME, np.float32), (i + 1) * STEP, 0.95)
            if item:
                outputs.append(item)
        for i in range(12):
            item = segmenter.feed(np.zeros(FRAME, np.float32), (211 + i) * STEP, 0.0)
            if item:
                outputs.append(item)
        self.assertGreaterEqual(len(outputs), 3)
        for item in outputs:
            self.assertEqual(evaluate(self.result, item, self.cfg)[1], 'continuous_speech')

    def test_vad_speech_end_excludes_silence_and_processing_time(self):
        segmenter = Segmenter(DEFAULTS)
        emitted = []
        for i in range(30):
            p = 0.9 if 5 <= i < 18 else 0
            item = segmenter.feed(np.ones(FRAME, np.float32), 100 + (i + 1) * STEP, p)
            if item:
                emitted.append(item)
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0].speech_end, 100 + 18 * STEP)
        self.assertGreater(emitted[0].voiced_seconds, 0.3)
        self.assertFalse(emitted[0].continued)

    def test_impulse_and_short_noise_do_not_emit(self):
        segmenter = Segmenter(DEFAULTS)
        for i in range(20):
            self.assertIsNone(segmenter.feed(np.zeros(FRAME, np.float32), i * STEP, 0.9 if i in (2, 7) else 0))

    def test_gap_does_not_join_fragments(self):
        segmenter = Segmenter(DEFAULTS)
        for i in range(6):
            segmenter.feed(np.zeros(FRAME, np.float32), i * STEP, 0.9)
        self.assertIsNone(segmenter.feed(np.zeros(FRAME, np.float32), 5, 0))
        self.assertIsNone(segmenter.flush())

    def test_deadline_boundary_cooldown_and_echo(self):
        gate = PlaybackGate(DEFAULTS)
        deadline = self.item.speech_end + DEFAULTS['max_latency'] - DEFAULTS['output_margin']
        self.assertIsNone(gate.reason(self.item, self.command, deadline - 0.001))
        self.assertEqual(gate.reason(self.item, self.command, deadline), 'deadline')
        gate.commit(self.item, self.command, 11.5, 12.8)
        self.assertEqual(gate.reason(self.item, self.command, 11.6), 'playback_overlap')
        fresh = Utterance(np.zeros(0), 13, 13.2, 0.2, uid=2)
        self.assertIsNone(gate.reason(fresh, self.command, 13.3))
        gate.last_played[self.command.name] = 13.1
        self.assertEqual(gate.reason(fresh, self.command, 13.3), 'cooldown')
        duplicate = Utterance(np.zeros(0), 14, 14.2, 0.2, uid=1)
        self.assertEqual(gate.reason(duplicate, self.command, 14.3), 'duplicate')

    def test_resampling_shape_and_antialiasing(self):
        original = np.sin(2 * np.pi * 12000 * np.arange(48000) / 48000).astype(np.float32)
        reduced = resample(original, 48000, SR)
        self.assertEqual(len(reduced), SR)
        self.assertLess(np.sqrt(np.mean(reduced[100:-100] ** 2)), 0.01)

    def test_settings_validation_and_disabled_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sound.wav').touch()
            data = {'commands': [dict(name='얼쑤', phrases=['얼쑤'], sounds=['sound.wav']),
                                 dict(name='좋다', enabled=False, phrases=['좋다'], sounds=['missing.wav'])]}
            path = root / 'commands.json'
            path.write_text(json.dumps(data))
            self.assertEqual(len(Config.load(path).commands), 1)
            for settings in [{'max_latency': 10.1}, {'vad_start': float('nan')},
                             {'start_frames': 1.5}, {'min_speech': True}, {'old_setting': 1}]:
                data['settings'] = settings
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    Config.load(path)


class OutputTests(unittest.TestCase):
    def player(self, now=10):
        # 실제 장치를 열지 않고 PortAudio callback의 동작을 검사한다.
        import threading
        player = Player.__new__(Player)
        player.lock = threading.Lock()
        player.notifications = queue.Queue(maxsize=32)
        player.cfg = SimpleNamespace(settings=dict(DEFAULTS))
        player.sr = SR
        player.pending = (np.ones((500, 1), np.float32),
                          Utterance(np.empty(0), 8, 9, 1, uid=1),
                          Command('얼쑤', ('얼쑤',), ()), 12.95)
        player.position = 0
        player.blocked_until = 0
        return player

    def test_deadline_checked_at_dac_not_only_at_schedule(self):
        player = self.player()
        out = np.empty((256, 1), np.float32)
        with patch('voice_trigger.time.monotonic', return_value=12.9):
            player.callback(out, 256, SimpleNamespace(outputBufferDacTime=1.1, currentTime=1), False)
        self.assertFalse(out.any())
        self.assertIsNone(player.pending)
        self.assertEqual(player.notifications.get()[0], 'deadline_at_output')

    def test_one_playback_event_and_exact_audio_tail(self):
        player = self.player()
        timing = SimpleNamespace(outputBufferDacTime=1.02, currentTime=1)
        out = np.empty((256, 1), np.float32)
        with patch('voice_trigger.time.monotonic', return_value=10):
            player.callback(out, 256, timing, False)
            self.assertTrue(np.all(out == 1))
            player.callback(out, 256, timing, False)
        self.assertTrue(np.all(out[:244] == 1))
        self.assertFalse(out[244:].any())
        self.assertIsNone(player.pending)
        self.assertEqual(player.notifications.qsize(), 1)
        self.assertEqual(player.notifications.get()[0], 'playback_started')

    def test_invalid_output_clock_fails_closed(self):
        player = self.player()
        out = np.ones((256, 1), np.float32)
        player.callback(out, 256, SimpleNamespace(outputBufferDacTime=math.nan, currentTime=1), False)
        self.assertFalse(out.any())
        self.assertEqual(player.notifications.get()[0], 'output_clock_error')


class WorkerTests(unittest.TestCase):
    def test_worker_contract_no_keyword_prompt_and_native_metadata(self):
        from unittest.mock import MagicMock
        connection = MagicMock()
        connection.recv.side_effect = [Utterance(np.zeros(SR, np.float32), 0, 1, 1, uid=7), None]
        result = {'text': '얼쑤', 'segments': []}
        engine = SimpleNamespace(transcribe=MagicMock(return_value=result))
        with patch.dict('sys.modules', {'mlx_whisper': engine}), patch('voice_trigger.local_model', return_value='local'):
            recognition_worker(connection, DEFAULTS)
        calls = connection.send.call_args_list
        self.assertEqual(calls[0].args[0][0], 'ready')
        self.assertEqual(calls[1].args[0][:2], ('result', 7))
        self.assertFalse(engine.transcribe.call_args.kwargs['word_timestamps'])
        self.assertIsNone(engine.transcribe.call_args.kwargs['initial_prompt'])
        self.assertFalse(engine.transcribe.call_args.kwargs['condition_on_previous_text'])


if __name__ == '__main__':
    unittest.main()
