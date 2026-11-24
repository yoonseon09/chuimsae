"""터미널 스페이스바 입력과 마이크 활성화 시간 제어. 추가 패키지를 사용하지 않는다."""
from __future__ import annotations

import math
import os
import select
import sys
import termios
import time
import tty

START_DELAY = 1.0
IDLE_TIMEOUT = 10.0


class MicActivation:
    """한 번 누르면 1초 후 열고, 발화가 끝나거나 10초 동안 말이 없으면 닫는다."""
    def __init__(self, silence):
        self.silence = silence
        self.close()

    def close(self):
        self.state = 'closed'
        self.ready_at = math.inf
        self.opened_at = None
        self.last_voice_end = None

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
        self.opened_at = now

    def voice(self, end):
        self.last_voice_end = end

    def close_reason(self, now, active):
        if self.state != 'listening':
            return None
        if self.last_voice_end is None:
            return 'no_speech_timeout' if now - self.opened_at >= IDLE_TIMEOUT else None
        if not active and now - self.last_voice_end >= self.silence:
            return 'speech_finished'
        return None


class TerminalSpaceKey:
    """Enter 없이 스페이스바를 읽는다. Ctrl+C와 종료 후 터미널 설정을 보존한다."""
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.fd = None
        self.saved = None
        self.last_space = -math.inf

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError('스페이스바 입력을 위해 일반 macOS 터미널에서 실행하세요.')
        self.fd = self.stream.fileno()
        self.saved = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd, termios.TCSANOW)
            # 모델 준비 중 눌렀던 키는 시작 명령으로 사용하지 않는다.
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except BaseException:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            self.saved = None
            raise
        return self

    def pressed(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return False
        data = os.read(self.fd, 1024)
        if not data:
            raise RuntimeError('터미널 입력이 닫혔습니다.')
        if b' ' not in data:
            return False
        now = time.monotonic()
        fresh = now - self.last_space >= 0.4
        self.last_space = now
        return fresh

    def __exit__(self, *args):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
            self.saved = None
