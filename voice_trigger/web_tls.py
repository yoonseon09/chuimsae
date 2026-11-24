"""Local HTTPS credentials. No OS trust store is changed automatically."""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import plistlib
import re
import socket
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlencode


def addresses():
    found = []
    try:
        output = subprocess.run(['/sbin/ifconfig'], capture_output=True, text=True, check=True).stdout
        for block in re.split(r'\n(?=\S)', output):
            # Only active physical network interfaces; avoid VPN/bridge endpoints.
            if not re.match(r'en\d+:', block) or 'status: active' not in block:
                continue
            for value in re.findall(r'\binet (\d+\.\d+\.\d+\.\d+)', block):
                if not ipaddress.ip_address(value).is_loopback:
                    found.append(value)
    except (OSError, subprocess.CalledProcessError):
        pass
    return list(dict.fromkeys(found))


def prepare_tls(folder: Path, hosts):
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    folder.chmod(0o700)
    ca, ca_key = folder / 'root.pem', folder / 'root.key'
    cert, cert_key = folder / 'server.pem', folder / 'server.key'
    hosts = sorted(set(['localhost', '127.0.0.1', *hosts]))
    for host in hosts:
        if not re.fullmatch(r'[a-zA-Z0-9.:-]+', host):
            raise ValueError('인증서 주소에는 IP 또는 영문 호스트 이름을 사용하세요.')

    def openssl(*args):
        subprocess.run(['/usr/bin/openssl', *map(str, args)], check=True, capture_output=True)

    ca_config = folder / 'ca.cnf'
    if not ca.exists() or not ca_key.exists():
        ca_config.write_text('[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ca\n'
                             '[dn]\nCN=Chuimsae Local Test Root\n[ca]\n'
                             'basicConstraints=critical,CA:TRUE,pathlen:0\n'
                             'keyUsage=critical,keyCertSign,cRLSign\n'
                             'subjectKeyIdentifier=hash\n', encoding='ascii')
        openssl('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-sha256', '-days', '365',
                '-keyout', ca_key, '-out', ca, '-config', ca_config)
    metadata = folder / 'hosts.json'
    valid = cert.exists() and cert_key.exists() and metadata.exists()
    if valid:
        try:
            valid = json.loads(metadata.read_text()) == hosts
            valid = valid and subprocess.run(['/usr/bin/openssl', 'x509', '-checkend', '86400',
                                             '-noout', '-in', str(cert)], capture_output=True).returncode == 0
        except (ValueError, OSError):
            valid = False
    if not valid:
        names = []
        for host in hosts:
            try:
                ipaddress.ip_address(host)
                names.append(f'IP:{host}')
            except ValueError:
                names.append(f'DNS:{host}')
        leaf_config = folder / 'server.cnf'
        leaf_config.write_text('[req]\nprompt=no\ndistinguished_name=dn\n[dn]\nCN=Chuimsae Local\n'
                               '[server]\nbasicConstraints=critical,CA:FALSE\n'
                               'keyUsage=critical,digitalSignature,keyEncipherment\n'
                               'extendedKeyUsage=serverAuth\nsubjectAltName=' + ','.join(names) + '\n', encoding='ascii')
        csr = folder / 'server.csr'
        openssl('req', '-new', '-newkey', 'rsa:2048', '-nodes', '-sha256', '-keyout', cert_key,
                '-out', csr, '-config', leaf_config)
        openssl('x509', '-req', '-in', csr, '-CA', ca, '-CAkey', ca_key, '-CAcreateserial',
                '-out', cert, '-days', '30', '-sha256', '-extfile', leaf_config, '-extensions', 'server')
        metadata.write_text(json.dumps(hosts))
    for path in (ca_key, cert_key):
        path.chmod(0o600)
    public_cert = folder / 'chuimsae-root.cer'
    openssl('x509', '-in', ca, '-outform', 'DER', '-out', public_cert)
    profile = {
        'PayloadType': 'Configuration', 'PayloadVersion': 1,
        'PayloadIdentifier': 'local.chuimsae.https', 'PayloadUUID': str(uuid.uuid4()),
        'PayloadDisplayName': '추임새 로컬 HTTPS',
        'PayloadDescription': '내 Mac의 추임새 테스트 페이지에 HTTPS로 연결하기 위한 인증서입니다. VPN이나 관리 기능은 포함하지 않습니다.',
        'PayloadContent': [{
            'PayloadType': 'com.apple.security.root', 'PayloadVersion': 1,
            'PayloadIdentifier': 'local.chuimsae.https.root', 'PayloadUUID': str(uuid.uuid4()),
            'PayloadDisplayName': 'Chuimsae Local Test Root',
            'PayloadContent': public_cert.read_bytes(),
        }],
    }
    (folder / 'chuimsae.mobileconfig').write_bytes(plistlib.dumps(profile))
    digest = hashlib.sha256(public_cert.read_bytes()).hexdigest().upper()
    fingerprint = ':'.join(digest[i:i + 2] for i in range(0, len(digest), 2))
    return cert, cert_key, fingerprint


def setup_html(host, port, token, fingerprint):
    query = urlencode({'key': token})
    target = html.escape(f'https://{host}:{port}/?{query}', quote=True)
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>추임새 · 휴대폰 연결</title>
<style>body{{font:16px/1.8 -apple-system,BlinkMacSystemFont,sans-serif;color:#34303e;margin:50px auto;padding:0 24px;max-width:570px;background:white}}h1{{letter-spacing:-1px}}h2{{font-size:20px;margin-top:30px}}a{{color:#8970b0;overflow-wrap:anywhere}}.button{{display:block;text-align:center;padding:13px;background:#f2edf9;border-radius:14px;text-decoration:none;margin:16px 0}}small{{display:block;color:#91879e;overflow-wrap:anywhere}}li{{margin:10px 0}}details{{margin:22px 0}}code{{overflow-wrap:anywhere}}</style>
<h1>목소리를 연결할 준비</h1><p>휴대폰과 Mac을 같은 Wi-Fi에 연결하세요. 처음 한 번, 이 Mac의 인증서를 휴대폰에 등록하면 마이크를 사용할 수 있어요.</p>
<h2>아이폰 · 아이패드</h2><ol><li><a href="/chuimsae.mobileconfig?{query}">추임새 인증서 프로파일 다운로드</a>를 누르고 허용하세요.</li><li>설정 → 일반 → VPN 및 기기 관리 → <b>추임새 로컬 HTTPS</b>를 설치하세요.</li><li>설정 → 일반 → 정보 → 인증서 신뢰 설정에서 <b>Chuimsae Local Test Root</b>의 신뢰를 켜세요.</li><li>아래 ‘추임새 열기’를 누르세요.</li></ol>
<details><summary>안드로이드 · Chrome</summary><ol><li><a href="/chuimsae-root.cer?{query}">CA 인증서 다운로드</a></li><li>설정의 ‘인증서 설치’에서 <b>CA 인증서</b>로 설치하세요. 보안 및 개인정보 보호 → 기타 보안 설정 → 인증서 설치에 있는 경우가 많습니다. 기종에 따라 메뉴 이름이 달라요.</li><li>아래 ‘추임새 열기’를 Chrome으로 여세요.</li></ol></details>
<a class="button" href="{target}">추임새 열기</a>
<p>원을 누른 채 말하고 손을 떼세요. 마이크 접근을 허용하는 동안 손을 뗐다면, 허용 후 다시 눌러주세요. 소리는 Mac에 연결된 스피커에서 나옵니다.</p>
<details><summary>인증서 확인과 테스트 후 삭제</summary><p>이 프로파일에는 인증서 1개만 포함되어 있습니다. Mac 터미널의 지문과 아래 값이 같은지 확인할 수 있어요.</p><small>SHA-256: {html.escape(fingerprint)}</small><p>테스트가 끝나면 아이폰의 VPN 및 기기 관리에서 ‘추임새 로컬 HTTPS’ 프로파일을 삭제하세요. 안드로이드는 사용자 인증서에서 삭제하세요.</p></details>
<p><small>인증서 설정 없이 경고 화면만 건너뛰면 마이크가 작동하지 않을 수 있습니다. Mac을 켜 두고 웹실행.command를 실행한 상태로 사용하세요.</small></p></html>'''
