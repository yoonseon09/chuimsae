"""터미널 스페이스바 입력과 마이크 활성화 시간 제어. 추가 패키지를 사용하지 않는다."""
from __future__ import annotations

import math
import ctypes
import os
import select
import sys
import termios
import tty

START_DELAY = 1.0


class MicActivation:
    """누른 채 1초 기다리면 열고, 손을 떼면 준비·입력을 즉시 종료한다."""
    def __init__(self):
        self.close()

    def close(self):
        self.state = 'closed'
        self.ready_at = math.inf

    def press(self, now):
        if self.state != 'closed':
            return False
        self.state = 'arming'
        self.ready_at = now + START_DELAY
        return True

    def due(self, now):
        return self.state == 'arming' and now >= self.ready_at

    def opened(self, now):
        self.state = 'listening'

    def release(self):
        previous = self.state
        self.close()
        return previous


class NativeSpaceState:
    """CoreGraphics의 실제 키 상태를 읽는다. 키 자동 반복 시간에는 의존하지 않는다."""
    def __init__(self):
        self.library = ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
        self.query = self.library.CGEventSourceKeyState
        self.query.argtypes = [ctypes.c_int32, ctypes.c_uint16]
        self.query.restype = ctypes.c_bool

    def __call__(self):
        # kCGEventSourceStateCombinedSessionState = 0, kVK_Space = 49
        return bool(self.query(0, 49))


class TerminalSpaceKey:
    """터미널에서 누름을 시작하고, 실제 키 상태로 유지·해제를 확인한다."""
    def __init__(self, stream=None, state_reader=None):
        self.stream = sys.stdin if stream is None else stream
        self.fd = None
        self.saved = None
        self.state_reader = state_reader
        self.latched = False
        self.wait_release = False

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError('스페이스바 입력을 위해 일반 macOS 터미널에서 실행하세요.')
        self.fd = self.stream.fileno()
        self.saved = termios.tcgetattr(self.fd)
        try:
            if self.state_reader is None:
                self.state_reader = NativeSpaceState()
            self.wait_release = self.state_reader()
            tty.setcbreak(self.fd, termios.TCSANOW)
            # 모델 준비 중 눌렀던 키는 시작 명령으로 사용하지 않는다.
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except BaseException:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            self.saved = None
            raise
        return self

    def held(self):
        space_received = False
        if select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 1024)
            if not data:
                raise RuntimeError('터미널 입력이 닫혔습니다.')
            space_received = b' ' in data
        down = self.state_reader()
        if not down:
            self.latched = False
            self.wait_release = False
        elif space_received and not self.wait_release:
            self.latched = True
        return bool(down and self.latched)

    def __exit__(self, *args):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            self.saved = None
