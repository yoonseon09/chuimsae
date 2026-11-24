"""마이크·STT·출력 장치만 대체해 실제 run_mic 제어 흐름을 검사한다."""
import queue
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from trigger_core import Command, DEFAULTS, STEP
from voice_trigger import run_mic
from benchmark import summarize


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, probabilities, inference_delay=0.064, input_overflow=False, voice_label=None,
                     texts=None, timeline=None):
        clock = SimpleNamespace(now=100.0)
        played = []
        counts = Counter()
        result = {'text': '얼쑤!', 'segments': [{
            'text': '얼쑤!', 'avg_logprob': -0.1, 'no_speech_prob': 0.01,
            'compression_ratio': 1.0, 'words': [{'word': '얼쑤!', 'probability': 0.95}]}]}
        cfg = SimpleNamespace(settings=dict(DEFAULTS), commands=tuple(
            Command(name, (name,), ()) for name in ('얼쑤', '잘한다', '좋다')))
        if voice_label:
            cfg.settings['voice_mode'] = 'model'
            cfg.commands = (Command('얼쑤', ('얼쑤',), (), voice_sounds={
                'male': (Path('male.wav'),), 'female': (Path('female.wav'),)}),)
            result['voice'] = {'label': voice_label, 'score': 0.95}

        class Frames:
            def __init__(self, maxsize):
                assert maxsize == 8
                self.index = 0

            def get(self, timeout):
                if self.index == len(probabilities):
                    raise KeyboardInterrupt()
                value = probabilities[self.index]
                self.index += 1
                clock.now = 100 + self.index * STEP
                return np.full(512, value, np.float32), clock.now

            def get_nowait(self):
                raise queue.Empty()

        class Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs['callback']

            def __enter__(self):
                if input_overflow:
                    self.callback(np.zeros((512, 1), np.float32), 512,
                                  SimpleNamespace(currentTime=1, inputBufferAdcTime=1), True)
                return self

            def __exit__(self, *args):
                return False

        class FakePlayer:
            def __init__(self, *args):
                self.notifications = queue.SimpleQueue()
                self.blocked_until = -float('inf')
                self.next_playback_at = -float('inf')

            def schedule(self, item, command, deadline=None):
                if clock.now < max(self.blocked_until, self.next_playback_at):
                    return None
                if clock.now >= deadline:
                    return None
                if voice_label:
                    assert command.sounds == (Path(voice_label + '.wav'),)
                played.append(item)
                if timeline is not None:
                    timeline.append((command.name, clock.now, item.uid))
                self.blocked_until = clock.now + 1.2
                self.next_playback_at = clock.now + 1.0 + cfg.settings['playback_gap']
                return self.blocked_until

            def close(self):
                pass

        class FakeRecognizer:
            current = None
            index = 0

            def send(self, item):
                assert self.current is None
                self.current = item
                self.until = clock.now + inference_delay

            def poll(self):
                if self.current is not None and clock.now + 1e-8 >= self.until:
                    item, self.current = self.current, None
                    output = result
                    if texts:
                        text = texts[min(self.index, len(texts) - 1)]
                        output = dict(result, text=text, segments=[dict(result['segments'][0], text=text)])
                    self.index += 1
                    return item, ('result', item.uid, output, inference_delay)
                return None

        events = SimpleNamespace(emit=lambda name, **fields: counts.update([name]))
        class FakeVAD:
            def reset(self):
                pass

            def __call__(self, data):
                return float(data[0])

        sd = SimpleNamespace(check_input_settings=lambda **kw: None,
                             InputStream=Stream, query_devices=lambda d: {'name': 'test'})
        with patch.dict('sys.modules', {'sounddevice': sd}), \
             patch('voice_trigger.resolve_device', return_value=0), \
             patch('voice_trigger.Player', FakePlayer), \
             patch('voice_trigger.queue.Queue', Frames), \
             patch('voice_trigger.time.monotonic', side_effect=lambda: clock.now), \
             patch('voice_trigger.log'):
            with self.assertRaises(KeyboardInterrupt):
                run_mic(cfg, events, FakeRecognizer(), FakeVAD())
        return played, counts

    def test_real_controller_ignores_speech_during_output_then_resumes(self):
        probabilities = ([0.9] * 16 + [0] * 20 + [0.9] * 16 + [0] * 200
                         + [0.9] * 16 + [0] * 24)
        played, counts = self.run_pipeline(probabilities)
        self.assertEqual(len(played), 2)
        self.assertGreater(played[1].start - played[0].speech_end, 1.2)

    def test_slow_recognition_preserves_pending_and_merges_repeats(self):
        probabilities = ([0.9] * 8 + [0] * 11) * 5 + [0] * 40
        played, counts = self.run_pipeline(probabilities, inference_delay=1.8)
        self.assertEqual(counts['overload_replaced'], 0)
        self.assertGreaterEqual(counts['repeated_interjection'], 1)
        # 다섯 발화의 인식이 끝날 때까지 기다린 후 한 번만 출력한다.
        played, counts = self.run_pipeline(probabilities + [0] * 280, inference_delay=1.8)
        self.assertEqual(len(played), 1)

    def test_expired_inference_never_schedules_output(self):
        probabilities = [0.9] * 16 + [0] * 360
        played, counts = self.run_pipeline(probabilities, inference_delay=10.1)
        self.assertFalse(played)
        self.assertEqual(counts['deadline'], 1)

    def test_seven_second_inference_can_play_with_ten_second_limit(self):
        played, counts = self.run_pipeline([0.9] * 16 + [0] * 260, inference_delay=7.0)
        self.assertEqual(len(played), 1)
        self.assertEqual(counts['deadline'], 0)

    def test_input_overflow_resets_and_recovers(self):
        played, counts = self.run_pipeline([0.9] * 16 + [0] * 20, input_overflow=True)
        self.assertEqual(counts['input_overflow'], 1)
        self.assertEqual(len(played), 1)

    def test_voice_result_selects_matching_audio_in_real_controller(self):
        for label in ('male', 'female'):
            played, counts = self.run_pipeline([0.9] * 16 + [0] * 24, voice_label=label)
            self.assertEqual(len(played), 1)

    def test_unknown_voice_never_schedules_output(self):
        played, counts = self.run_pipeline([0.9] * 16 + [0] * 20, voice_label='unknown')
        self.assertFalse(played)

    def test_sentence_with_keyword_schedules_audio(self):
        timeline = []
        played, counts = self.run_pipeline([.9] * 40 + [0] * 40,
            texts=['아 이거 되는건가 얼쑤?'], timeline=timeline)
        self.assertEqual([row[0] for row in timeline], ['얼쑤'])

    def test_three_repetitions_in_one_sentence_only_play_once(self):
        timeline = []
        self.run_pipeline([.9] * 50 + [0] * 300,
                          texts=['얼쑤 얼쑤 얼쑤'], timeline=timeline)
        self.assertEqual([row[0] for row in timeline], ['얼쑤'])

    def test_separate_repeated_utterances_only_play_once(self):
        timeline = []
        played, counts = self.run_pipeline(([.9] * 10 + [0] * 12) * 3 + [0] * 350,
                          texts=['얼쑤', '얼쑤', '얼쑤'], timeline=timeline)
        self.assertEqual([row[0] for row in timeline], ['얼쑤'])
        self.assertEqual(counts['repeated_interjection'], 2)

    def test_distinct_words_in_sentence_wait_two_seconds_after_audio_end(self):
        timeline = []
        self.run_pipeline([.9] * 50 + [0] * 400,
                          texts=['얼쑤 잘한다 좋다'], timeline=timeline)
        self.assertEqual([row[0] for row in timeline], ['얼쑤', '잘한다', '좋다'])
        self.assertTrue(all(b[1] - a[1] >= 3.0 for a, b in zip(timeline, timeline[1:])))
        self.assertEqual(len({row[2] for row in timeline}), 1)

    def test_separate_distinct_utterances_are_not_replaced_under_slow_recognition(self):
        timeline = []
        self.run_pipeline(([.9] * 10 + [0] * 12) * 3 + [0] * 500,
                          inference_delay=1.8, texts=['얼쑤', '잘한다', '좋다'], timeline=timeline)
        self.assertEqual([row[0] for row in timeline], ['얼쑤', '잘한다', '좋다'])
        self.assertTrue(all(b[1] - a[1] >= 3.0 for a, b in zip(timeline, timeline[1:])))

    def test_new_distinct_command_during_silent_output_gap_is_captured(self):
        timeline = []
        self.run_pipeline([.9] * 10 + [0] * 60 + [.9] * 10 + [0] * 230,
                          texts=['얼쑤', '좋다'], timeline=timeline)
        self.assertEqual([row[0] for row in timeline], ['얼쑤', '좋다'])
        self.assertGreaterEqual(timeline[1][1] - timeline[0][1], 3.0)

    def test_long_continuous_sentence_is_not_dropped_at_chunk_boundary(self):
        timeline = []
        self.run_pipeline([.9] * 220 + [0] * 400,
                          texts=['이 문장은 길지만 얼쑤', '그 다음 잘한다 좋다'], timeline=timeline)
        self.assertEqual([row[0] for row in timeline], ['얼쑤', '잘한다', '좋다'])

    def test_benchmark_counts_duplicates_and_negatives(self):
        rows = [dict(expected='얼쑤', predicted=['얼쑤', '얼쑤'], duration_seconds=1, inference_seconds=[0.2]),
                dict(expected='얼쑤', predicted=[], duration_seconds=1, inference_seconds=[]),
                dict(expected=None, predicted=['얼쑤'], duration_seconds=60, inference_seconds=[0.3])]
        report = summarize(rows)
        self.assertEqual(report['recall'], 0.5)
        self.assertEqual(report['candidate_precision'], 1 / 3)
        self.assertEqual(report['false_candidate_events_per_negative_hour'], 60)
        self.assertEqual(report['extra_or_wrong_candidate_events'], 1)
        self.assertFalse(report['actual_output_latency_measured'])


if __name__ == '__main__':
    unittest.main()
