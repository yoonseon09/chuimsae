import io
import os
import pty
import termios
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mic_control import MicActivation, NativeSpaceState, TerminalSpaceKey


class MicActivationTests(unittest.TestCase):
    def test_space_opens_only_after_one_second(self):
        mic = MicActivation()
        self.assertEqual(mic.state, 'closed')
        self.assertTrue(mic.press(10))
        self.assertFalse(mic.due(10.999))
        self.assertTrue(mic.due(11))
        self.assertFalse(mic.press(10.5))  # 반복 키로 대기 시간을 연장하지 않는다.
        mic.opened(11)
        self.assertFalse(mic.press(12))

    def test_release_closes_listening_and_rearming_gets_full_one_second(self):
        mic = MicActivation()
        mic.press(10)
        mic.opened(11)
        self.assertEqual(mic.release(), 'listening')
        self.assertEqual(mic.state, 'closed')
        self.assertTrue(mic.press(13))
        self.assertEqual(mic.ready_at, 14)

    def test_release_before_one_second_cancels_activation(self):
        mic = MicActivation()
        mic.press(10)
        self.assertEqual(mic.release(), 'arming')
        self.assertFalse(mic.due(11))
        self.assertEqual(mic.state, 'closed')

    def test_native_reader_checks_space_key_code_and_current_session(self):
        library = SimpleNamespace(CGEventSourceKeyState=MagicMock(return_value=True))
        with patch('mic_control.ctypes.CDLL', return_value=library):
            reader = NativeSpaceState()
        self.assertTrue(reader())
        library.CGEventSourceKeyState.assert_called_once_with(0, 49)


class TerminalSpaceKeyTests(unittest.TestCase):
    def test_space_without_enter_and_terminal_restore_even_on_exception(self):
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        original = termios.tcgetattr(stream.fileno())
        state = SimpleNamespace(down=False)
        try:
            with self.assertRaisesRegex(ValueError, 'test exit'):
                with TerminalSpaceKey(stream, state_reader=lambda: state.down) as key:
                    attributes = termios.tcgetattr(stream.fileno())
                    self.assertFalse(attributes[3] & termios.ICANON)
                    self.assertFalse(attributes[3] & termios.ECHO)
                    self.assertTrue(attributes[3] & termios.ISIG)
                    self.assertFalse(key.held())
                    os.write(master, b'x')
                    self.assertFalse(key.held())
                    state.down = True
                    os.write(master, b' ')
                    self.assertTrue(key.held())
                    # 반복 문자가 없어도 누른 상태를 계속 유지한다.
                    self.assertTrue(key.held())
                    os.write(master, b' ')
                    self.assertTrue(key.held())
                    state.down = False
                    self.assertFalse(key.held())  # 키를 뗄 때 터미널 문자는 필요 없다.
                    os.write(master, b' ')
                    self.assertFalse(key.held())  # 이미 떼었던 키의 남은 문자는 무시
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

    def test_key_held_during_startup_requires_release_and_fresh_terminal_space(self):
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        state = SimpleNamespace(down=True)
        try:
            with TerminalSpaceKey(stream, state_reader=lambda: state.down) as key:
                os.write(master, b' ')
                self.assertFalse(key.held())
                state.down = False
                self.assertFalse(key.held())
                state.down = True
                self.assertFalse(key.held())  # 다른 창에서 누른 키로 시작하지 않는다.
                os.write(master, b' ')
                self.assertTrue(key.held())
        finally:
            stream.close()
            os.close(master)

    def test_non_terminal_fails_with_clear_message(self):
        with self.assertRaisesRegex(RuntimeError, '터미널'):
            with TerminalSpaceKey(io.StringIO()):
                pass


if __name__ == '__main__':
    unittest.main()
