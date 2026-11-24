import json
import math
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from trigger_core import (Command, Config, DEFAULTS, FRAME, SR, STEP,
                          PlaybackQueue, Segmenter, Utterance, evaluate, evaluate_commands,
                          normalize, resolve_path, text_command, text_commands)
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

    def test_registered_word_inside_sentence(self):
        self.assertEqual(text_command(' 얼쑤! ', self.cfg.commands), self.command)
        self.assertEqual(text_command('아 이거 되는건가 얼쑤?', self.cfg.commands), self.command)
        for text in ['얼쑤라고 말해 봐', '어서 오세요', '얼쑥', '좋다', '얼쑤a', '얼쑤1', '안얼쑤', '']:
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

    def test_long_sentence_fragments_are_now_eligible(self):
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
        self.assertGreaterEqual(len(outputs), 2)
        for item in outputs:
            self.assertEqual(evaluate(self.result, item, self.cfg)[1], 'accepted')

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
                             {'start_frames': 1.5}, {'min_speech': True}, {'old_setting': 1},
                             {'queue_capacity': 1.5}, {'queue_capacity': True},
                             {'duplicate_window': -1}, {'playback_gap': float('nan')},
                             {'queue_timeout': 0}, {'burst_silence': 0}]:
                data['settings'] = settings
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    Config.load(path)


class SequencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.commands = tuple(Command(name, (name,), ()) for name in ('얼쑤', '좋다', '잘한다'))
        self.queue = PlaybackQueue(DEFAULTS)

    def item(self, uid, start, end=None):
        return Utterance(np.empty(0), start, start + .4 if end is None else end, .4, uid=uid)

    def test_sentence_order_is_not_configuration_order(self):
        names = lambda text: [c.name for c in text_commands(text, self.commands)]
        self.assertEqual(names('아 이거 되는건가 얼쑤?'), ['얼쑤'])
        self.assertEqual(names('얼쑤, 잘한다, 좋다!'), ['얼쑤', '잘한다', '좋다'])
        self.assertEqual(names('얼쑤 얼쑤 얼쑤 얼쑤'), ['얼쑤'])
        self.assertEqual(names('얼쑤얼쑤얼쑤'), ['얼쑤'])
        self.assertEqual(names('얼쑤! 잘한다! 얼쑤! 좋다!'), ['얼쑤', '잘한다', '좋다'])
        self.assertEqual(names('얼 쑤, 잘 한다, 좋다'), ['얼쑤', '잘한다', '좋다'])
        self.assertEqual(names('얼쑤라고 했고 잘한다는 말과 안좋다는 말'), [])
        import unicodedata
        self.assertEqual(names(unicodedata.normalize('NFD', '자 얼쑤!')), ['얼쑤'])

    def test_legacy_exact_setting_remains_available(self):
        exact = Command('얼쑤', ('얼쑤',), (), match='exact')
        self.assertEqual(text_commands('얼 쑤!', (exact,)), [exact])
        self.assertEqual(text_commands('자 얼쑤!', (exact,)), [])

    def test_all_candidates_pass_through_quality_gate(self):
        text = '얼쑤, 잘한다, 좋다'
        result = dict(text=text, segments=[dict(text=text, avg_logprob=-.1,
                      no_speech_prob=.01, compression_ratio=1)])
        cfg = SimpleNamespace(settings=DEFAULTS, commands=self.commands)
        commands, reason = evaluate_commands(result, self.item(1, 0), cfg)
        self.assertEqual([c.name for c in commands], ['얼쑤', '잘한다', '좋다'])
        result['segments'][0]['no_speech_prob'] = .99
        self.assertEqual(evaluate_commands(result, self.item(1, 0), cfg), ([], 'no_speech'))

    def test_repeated_inputs_merge_even_with_delayed_inference(self):
        command = self.commands[0]
        self.assertIsNone(self.queue.offer(self.item(1, 0), command, 7))
        self.queue.pending.popleft()  # 첫 출력 이후에도 입력 시각으로 중복 판단
        self.assertEqual(self.queue.offer(self.item(2, 1), command, 9), 'repeated_interjection')
        self.assertEqual(self.queue.offer(self.item(3, 2), command, 11), 'repeated_interjection')
        self.assertFalse(self.queue.pending)
        self.assertIsNone(self.queue.offer(self.item(4, 7.5), command, 8))

    def test_pending_command_merges_until_played_even_after_five_seconds(self):
        command = self.commands[0]
        self.assertIsNone(self.queue.offer(self.item(1, 0), command, 1))
        self.assertEqual(self.queue.offer(self.item(2, 8), command, 9), 'repeated_interjection')
        self.assertEqual(len(self.queue.pending), 1)

    def test_same_utterance_can_enqueue_all_distinct_commands_in_order(self):
        item = self.item(1, 0)
        for command in text_commands('얼쑤 잘한다 좋다', self.commands):
            self.assertIsNone(self.queue.offer(item, command, 1))
        self.assertEqual([r.command.name for r in self.queue.pending], ['얼쑤', '잘한다', '좋다'])
        self.assertEqual(self.queue.offer(item, self.commands[0], 2), 'duplicate')
        self.assertEqual(len(self.queue.expire(61)), 3)

    def test_queue_capacity_and_recognition_deadline(self):
        q = PlaybackQueue(dict(DEFAULTS, queue_capacity=1))
        item = self.item(1, 0)
        self.assertEqual(q.offer(item, self.commands[0], 11), 'deadline')
        self.assertIsNone(q.offer(item, self.commands[0], 1))
        self.assertEqual(q.offer(item, self.commands[1], 1), 'playback_queue_full')


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

    def test_real_scheduler_gap_uses_dac_end_and_allows_queued_wait_over_ten_seconds(self):
        player = self.player()
        player.dry = False
        player.stream = SimpleNamespace(latency=.1)
        player.pending = None
        player.next_playback_at = -math.inf
        commands = tuple(Command(name, (name,), (Path(name + '.wav'),))
                         for name in ('얼쑤', '잘한다', '좋다'))
        # 각 음원은 4초. 세 번째 출력은 원래 발화 후 10초를 넘어야 한다.
        player.cache = {(c.name, c.sounds[0]): np.ones((SR * 4, 1), np.float32) for c in commands}
        item = Utterance(np.empty(0), 8, 9, 1, uid=1)
        for index, command in enumerate(commands):
            start = 10 + index * 6.05
            with patch('voice_trigger.time.monotonic', return_value=start):
                self.assertIsNotNone(player.schedule(item, command, deadline=69))
            out = np.empty((SR * 4, 1), np.float32)
            with patch('voice_trigger.time.monotonic', return_value=start):
                player.callback(out, len(out), SimpleNamespace(
                    outputBufferDacTime=1.05, currentTime=1), False)
            self.assertTrue(out.all())
            self.assertIsNone(player.pending)
            self.assertAlmostEqual(player.next_playback_at, start + 4.05 + 2)
            if index < 2:
                with patch('voice_trigger.time.monotonic', return_value=player.next_playback_at - .001):
                    self.assertIsNone(player.schedule(item, commands[index + 1], deadline=69))

    def test_dry_run_scheduler_also_preserves_two_second_gap(self):
        player = self.player()
        player.dry = True
        player.pending = None
        player.blocked_until = -math.inf
        player.next_playback_at = -math.inf
        command = Command('얼쑤', ('얼쑤',), (Path('sound.wav'),))
        player.cache = {(command.name, command.sounds[0]): np.ones((SR, 1), np.float32)}
        item = Utterance(np.empty(0), 8, 9, 1, uid=1)
        with patch('voice_trigger.time.monotonic', return_value=10):
            self.assertIsNotNone(player.schedule(item, command, deadline=69))
        self.assertAlmostEqual(player.blocked_until, 11.2)
        self.assertAlmostEqual(player.next_playback_at, 13)
        with patch('voice_trigger.time.monotonic', return_value=12.999):
            self.assertIsNone(player.schedule(item, command, deadline=69))
        with patch('voice_trigger.time.monotonic', return_value=13):
            self.assertIsNotNone(player.schedule(item, command, deadline=69))


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
