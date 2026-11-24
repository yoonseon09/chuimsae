import io
import os
import pty
import termios
import unittest
from unittest.mock import patch

from mic_control import MicActivation, TerminalSpaceKey


class MicActivationTests(unittest.TestCase):
    def test_space_opens_only_after_one_second(self):
        mic = MicActivation(.6)
        self.assertEqual(mic.state, 'closed')
        self.assertTrue(mic.press(10))
        self.assertFalse(mic.due(10.999))
        self.assertTrue(mic.due(11))
        self.assertFalse(mic.press(10.5))  # 반복 키로 대기 시간을 연장하지 않는다.
        mic.opened(11)
        self.assertFalse(mic.press(12))

    def test_burst_silence_closes_only_after_speech_ends(self):
        mic = MicActivation(.6)
        mic.press(10)
        mic.opened(11)
        mic.voice(12)
        self.assertIsNone(mic.close_reason(12.5, False))
        self.assertIsNone(mic.close_reason(12.7, True))
        self.assertEqual(mic.close_reason(12.7, False), 'speech_finished')
        mic.close()
        self.assertTrue(mic.press(13))
        self.assertEqual(mic.ready_at, 14)
        self.assertIsNone(mic.last_voice_end)

    def test_no_voice_timeout_starts_from_actual_open_time(self):
        mic = MicActivation(.6)
        mic.press(0)
        self.assertIsNone(mic.close_reason(100, False))
        mic.opened(100)
        self.assertIsNone(mic.close_reason(109.999, False))
        self.assertEqual(mic.close_reason(110, False), 'no_speech_timeout')


class TerminalSpaceKeyTests(unittest.TestCase):
    def test_space_without_enter_and_terminal_restore_even_on_exception(self):
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        original = termios.tcgetattr(stream.fileno())
        try:
            with self.assertRaisesRegex(ValueError, 'test exit'):
                with TerminalSpaceKey(stream) as key:
                    attributes = termios.tcgetattr(stream.fileno())
                    self.assertFalse(attributes[3] & termios.ICANON)
                    self.assertFalse(attributes[3] & termios.ECHO)
                    self.assertTrue(attributes[3] & termios.ISIG)
                    self.assertFalse(key.pressed())
                    os.write(master, b'x')
                    self.assertFalse(key.pressed())
                    os.write(master, b' ')
                    with patch('mic_control.time.monotonic', return_value=10):
                        self.assertTrue(key.pressed())
                    os.write(master, b' ')
                    with patch('mic_control.time.monotonic', return_value=10.05):
                        self.assertFalse(key.pressed())
                    os.write(master, b' ')
                    with patch('mic_control.time.monotonic', return_value=11):
                        self.assertTrue(key.pressed())
                    raise ValueError('test exit')
            restored = termios.tcgetattr(stream.fileno())
            # macOS 커널은 모드 복구 때 PENDIN(입력 재표시 예정)을 자동으로 세운다.
            # 동작 설정과 제어 문자는 이 일시적 상태 비트를 제외하고 모두 같아야 한다.
            for attributes in (restored, original):
                attributes[3] &= ~getattr(termios, 'PENDIN', 0)
            self.assertEqual(restored, original)
        finally:
            stream.close()
            os.close(master)

    def test_non_terminal_fails_with_clear_message(self):
        with self.assertRaisesRegex(RuntimeError, '터미널'):
            with TerminalSpaceKey(io.StringIO()):
                pass


if __name__ == '__main__':
    unittest.main()
