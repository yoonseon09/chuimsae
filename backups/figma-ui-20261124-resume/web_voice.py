#!/usr/bin/env python3
"""HTTPS phone UI → held-only PCM streaming → local Mac recognition and output."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import math
import secrets
import socket
import ssl
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from web_backend import BUDGET, MAX_CAPTURE, Backend
from web_tls import addresses, prepare_tls, setup_html

HERE = Path(__file__).resolve().parent
WEB = HERE / 'web'


class ProtocolError(ValueError):
    pass


class Capture:
    def __init__(self, data, now):
        self.id = data.get('id')
        self.rate = data.get('sample_rate')
        if not isinstance(self.id, str) or not 1 <= len(self.id) <= 80:
            raise ProtocolError('입력 ID가 올바르지 않아요.')
        if type(self.rate) is not int or not 8000 <= self.rate <= 96000:
            raise ProtocolError('마이크 샘플 속도를 확인해주세요.')
        self.begun = now
        self.pcm = bytearray()

    def append(self, data, now):
        if not data or len(data) % 2 or len(data) > 65536:
            raise ProtocolError('음성 데이터 형식이 올바르지 않아요.')
        if len(self.pcm) + len(data) > int(self.rate * 2 * MAX_CAPTURE) or now - self.begun > 15:
            raise ProtocolError('한 번에 최대 10초까지 말해주세요.')
        self.pcm.extend(data)

    def finish(self, data, now, uid):
        if data.get('id') != self.id or type(data.get('samples')) is not int or data['samples'] != len(self.pcm) // 2:
            raise ProtocolError('음성 일부가 도착하지 않아 입력을 취소했어요.')
        if len(self.pcm) < self.rate * 2 * 0.1:
            raise ProtocolError('입력이 너무 짧아요. 원을 누른 채 말해주세요.')
        released_ms = data.get('released_server_ms')
        uncertainty = data.get('clock_uncertainty_ms')
        synced = (type(released_ms) in (int, float) and math.isfinite(released_ms)
                  and now - 12 <= released_ms / 1000 <= now + 0.1
                  and type(uncertainty) in (int, float) and math.isfinite(uncertainty)
                  and 0 <= uncertainty <= 500)
        # Subtract the clock uncertainty: a result cannot gain extra time from sync error.
        released = min(now, (released_ms - uncertainty) / 1000) if synced else now
        return dict(pcm=bytes(self.pcm), sample_rate=self.rate, released_at=released,
                    uid=uid, timing_synced=synced, clock_uncertainty_ms=uncertainty if synced else None)


class VoiceService:
    def __init__(self, config_path, token, preview=False):
        self.config_path, self.token, self.preview = config_path, token, preview
        self.state = 'ready' if preview else 'warming'
        self.detail = '디자인 미리보기입니다. 인식과 음원 출력은 하지 않습니다.' if preview else 'Mac의 음성 인식·목소리 분류 모델을 준비하고 있어요.'
        self.backend = None
        self.clients = set()
        self.owner = None
        self.phase = None
        self.capture = None
        self.uid = 0
        self.task = None
        self.work_started = None
        self.released_at = None
        self.timed_out = False
        self.played = False

    def status(self):
        return dict(type='status', state=self.state, phase=self.phase, detail=self.detail,
                    mode='preview' if self.preview else 'live')

    async def broadcast(self):
        for client in tuple(self.clients):
            if not client.closed:
                with contextlib.suppress(ConnectionError, RuntimeError):
                    await client.send_json(self.status())

    async def tell_owner(self, message):
        if self.owner is not None and not self.owner.closed:
            with contextlib.suppress(ConnectionError, RuntimeError):
                await self.owner.send_json(message)

    async def start(self, app):
        if not self.preview:
            self.backend = Backend(self.config_path)
        self.task = asyncio.create_task(self.monitor())

    async def stop(self, app):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        for client in tuple(self.clients):
            await client.close(code=1001, message=b'Mac server stopped')
        if self.backend:
            self.backend.close()

    async def ready(self):
        self.state, self.phase, self.capture, self.owner = 'ready', None, None, None
        self.work_started = None
        self.detail = ('디자인과 음성 전송만 확인합니다. 인식과 재생은 하지 않습니다.' if self.preview
                       else '원을 누른 채 말하고 손을 떼세요. 소리는 Mac 스피커에서 재생됩니다.')
        await self.broadcast()

    async def monitor(self):
        while True:
            now = time.monotonic()
            if self.phase == 'capturing' and self.capture and now - self.capture.begun > 15:
                await self.tell_owner(dict(type='error', detail='입력 시간이 길어 마이크 연결을 종료했어요.'))
                await self.ready()
            if self.backend:
                try:
                    while self.backend.connection.poll():
                        message = self.backend.connection.recv()
                        kind = message['type']
                        if kind == 'backend_ready':
                            await self.ready()
                        elif kind == 'backend_error':
                            self.state = 'error'
                            self.detail = 'Mac의 모델 또는 출력 장치를 준비하지 못했어요: ' + message['detail']
                            print('\n[Mac 준비 실패] ' + self.detail, flush=True)
                            await self.broadcast()
                            self.backend.close(); self.backend = None
                            break
                        else:
                            if kind == 'playback':
                                self.played = True
                                print(f"[출력 예정] {message['command']} | 말 끝부터 약 {message['latency_ms'] / 1000:.3f}초 (DAC 추정)", flush=True)
                            if not self.timed_out:
                                await self.tell_owner(message)
                    if self.backend and not self.backend.process.is_alive():
                        raise RuntimeError('음성 처리 프로세스가 종료됐어요.')
                except (EOFError, OSError, RuntimeError) as exc:
                    self.state, self.detail = 'error', str(exc)
                    await self.broadcast()
                    if self.backend:
                        self.backend.close(); self.backend = None
            if self.phase == 'processing' and self.work_started:
                if not self.played and not self.timed_out and now >= self.released_at + BUDGET:
                    self.timed_out = True
                    if self.backend:
                        self.backend.cancelled.set()
                    await self.tell_owner(dict(type='outcome', kind='error', title='3초 안에 처리하지 못했어요',
                                               detail='늦어진 음원은 재생하지 않아요. Mac의 인식 속도를 확인해주세요.'))
                if now - self.work_started > 20:
                    self.state = 'error'
                    self.detail = 'Mac의 음성 처리가 응답하지 않아요. 웹실행.command를 다시 실행해주세요.'
                    self.phase = None
                    if self.backend:
                        self.backend.close(); self.backend = None
                    await self.broadcast()
            await asyncio.sleep(0.015)

    async def abandon(self, client):
        if self.owner is not client:
            return
        if self.phase == 'capturing':
            await self.ready()
        elif self.phase == 'processing':
            if self.backend:
                self.backend.cancelled.set()
            self.owner = None

    async def websocket(self, request):
        origin = request.headers.get('Origin')
        if origin and origin != f'{request.scheme}://{request.host}':
            raise web.HTTPForbidden(text='다른 페이지에서는 연결할 수 없습니다.')
        ws = web.WebSocketResponse(heartbeat=15, max_msg_size=65536, receive_timeout=30)
        await ws.prepare(request)
        authenticated = False
        try:
            first = await asyncio.wait_for(ws.receive_json(), timeout=5)
            if not isinstance(first, dict) or first.get('type') != 'auth' or not self.authorized(first.get('key')):
                await ws.close(code=1008, message=b'Open the URL shown on your Mac')
                return ws
            self.clients.add(ws)
            authenticated = True
            await ws.send_json(self.status())
            async for message in ws:
                try:
                    now = time.monotonic()
                    if message.type == WSMsgType.BINARY:
                        if self.owner is not ws or self.phase != 'capturing' or not self.capture:
                            raise ProtocolError('녹음을 시작한 다음 음성을 보내주세요.')
                        self.capture.append(message.data, now)
                    elif message.type == WSMsgType.TEXT:
                        data = json.loads(message.data)
                        if not isinstance(data, dict):
                            raise ProtocolError('요청 형식이 올바르지 않아요.')
                        kind = data.get('type')
                        if kind == 'clock':
                            value = data.get('client_ms')
                            if type(value) not in (int, float) or not math.isfinite(value):
                                raise ProtocolError('연결 시간을 확인하지 못했어요.')
                            await ws.send_json(dict(type='clock', client_ms=value, server_ms=now * 1000))
                        elif kind == 'begin':
                            if self.state != 'ready':
                                await ws.send_json(dict(type='rejected', detail='Mac이 준비될 때까지 잠시 기다려주세요.'))
                                continue
                            capture = Capture(data, now)
                            self.owner, self.capture, self.phase, self.state = ws, capture, 'capturing', 'busy'
                            self.detail = '휴대폰의 목소리를 받고 있어요.'
                            await ws.send_json(dict(type='capture_ready', id=capture.id))
                            await self.broadcast()
                        elif kind == 'end':
                            if self.owner is not ws or self.phase != 'capturing':
                                raise ProtocolError('진행 중인 녹음이 없어요.')
                            self.uid += 1
                            job = self.capture.finish(data, now, self.uid)
                            self.capture = None
                            if self.preview:
                                await ws.send_json(dict(type='outcome', kind='preview', title='목소리가 Mac에 도착했어요',
                                                        detail=f'{len(job["pcm"]) / (2 * job["sample_rate"]):.1f}초 입력 · 미리보기에서는 인식과 재생을 하지 않아요.'))
                                await self.ready()
                            else:
                                self.phase, self.work_started = 'processing', now
                                self.released_at, self.timed_out, self.played = job['released_at'], False, False
                                self.detail = 'Mac에서 음성을 인식하고 있어요.'
                                self.backend.cancelled.clear()
                                self.backend.connection.send(job)
                                await self.broadcast()
                        elif kind == 'cancel':
                            await self.abandon(ws)
                        else:
                            raise ProtocolError('지원하지 않는 요청이에요.')
                    elif message.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        break
                except (ProtocolError, ValueError, TypeError, KeyError) as exc:
                    await ws.send_json(dict(type='error', detail=str(exc)[:300]))
                    await self.abandon(ws)
        except asyncio.TimeoutError:
            await ws.close(code=1001 if authenticated else 1008, message=b'Connection timed out')
        except (ValueError, TypeError):
            await ws.close(code=1008, message=b'Authentication required')
        finally:
            self.clients.discard(ws)
            await self.abandon(ws)
        return ws

    def authorized(self, candidate):
        return isinstance(candidate, str) and hmac.compare_digest(candidate.encode('utf-8'), self.token.encode('utf-8'))


@web.middleware
async def headers(request, handler):
    response = await handler(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Permissions-Policy'] = 'microphone=(self), camera=()'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    return response


def create_app(service):
    app = web.Application(middlewares=[headers], client_max_size=65536)
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)
    app.router.add_get('/ws', service.websocket)

    async def health(request):
        return web.json_response(service.status())

    app.router.add_get('/health', health)
    for name in ('index.html', 'style.css', 'app.js', 'pcm-worklet.js', 'icon.svg',
                 'fonts/MaruBuri-Regular.woff2', 'fonts/MaruBuri-SemiBold.woff2',
                 'fonts/MaruBuri-Bold.woff2'):
        async def asset(request, filename=name):
            return web.FileResponse(WEB / filename)
        app.router.add_get('/' if name == 'index.html' else '/' + name, asset)
    return app


def create_setup_app(service, folder, host, port, fingerprint):
    async def setup(request):
        if not service.authorized(request.query.get('key')):
            raise web.HTTPForbidden(text='Mac 터미널에 나온 휴대폰 설정 주소를 열어주세요.')
        filename = request.match_info.get('filename')
        if filename:
            response = web.FileResponse(folder / filename)
            response.content_type = 'application/x-apple-aspen-config' if filename.endswith('.mobileconfig') else 'application/x-x509-ca-cert'
            return response
        return web.Response(text=setup_html(host, port, service.token, fingerprint), content_type='text/html')

    app = web.Application()
    app.router.add_get('/', setup)
    app.router.add_get('/{filename:chuimsae\\.mobileconfig|chuimsae-root\\.cer}', setup)
    return app


async def serve(args):
    token = secrets.token_urlsafe(24)
    service = VoiceService(args.config, token, preview=args.preview)
    found = addresses()
    host = args.address or (found[0] if found else 'localhost')
    hostname = socket.gethostname()
    folder = HERE / '.web-local'
    cert, cert_key, fingerprint = prepare_tls(folder, [host, hostname, *found])
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, cert_key)
    runner = web.AppRunner(create_app(service), access_log=None)
    setup_runner = web.AppRunner(create_setup_app(service, folder, host, args.port, fingerprint), access_log=None)
    try:
        await runner.setup()
        await setup_runner.setup()
        await web.TCPSite(runner, '0.0.0.0', args.port, ssl_context=context).start()
        await web.TCPSite(setup_runner, '0.0.0.0', args.setup_port).start()
        print('\n추임새 · 휴대폰 → Mac 스피커' + (' [디자인 미리보기]' if args.preview else ''), flush=True)
        print(f'처음 한 번 휴대폰 설정: http://{host}:{args.setup_port}/?key={token}', flush=True)
        print(f'휴대폰 접속: https://{host}:{args.port}/?key={token}', flush=True)
        print(f'Mac 화면: https://localhost:{args.port}/?key={token}', flush=True)
        print(f'인증서 SHA-256: {fingerprint}', flush=True)
        print('같은 Wi-Fi에 연결한 휴대폰에서 설정 주소를 여세요. 종료: Ctrl+C\n', flush=True)
        await asyncio.Event().wait()
    finally:
        await setup_runner.cleanup()
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'commands.json')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--setup-port', type=int, default=8766)
    parser.add_argument('--address', help='휴대폰에서 접속할 Mac의 Wi-Fi IPv4 주소')
    parser.add_argument('--preview', action='store_true', help='모델·스피커 없이 디자인과 음성 전송만 미리보기')
    args = parser.parse_args()
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        print('\n웹 서버를 종료했습니다.')


if __name__ == '__main__':
    main()
