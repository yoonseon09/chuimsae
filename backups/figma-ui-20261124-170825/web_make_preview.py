"""Build a self-contained design review file; it never accesses the microphone."""
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent


def build():
    page = (HERE / 'web/index.html').read_text()
    style = (HERE / 'web/style.css').read_text()
    icon = quote((HERE / 'web/icon.svg').read_text(), safe='')
    page = page.replace('<link rel="stylesheet" href="/style.css">', f'<style>{style}</style>')
    page = page.replace('href="/icon.svg"', f'href="data:image/svg+xml,{icon}"')
    page = page.replace('<script type="module" src="/app.js"></script>', '')
    page = page.replace('href="/"', 'href="#"')
    page = page.replace('<title>추임새 — 목소리에 응답하는 순간</title>', '<title>추임새 · 디자인 미리보기</title>')
    page = page.replace('</body>', '''<script>
const el = (id) => document.getElementById(id);
function ready() {
  document.body.dataset.state = 'ready';
  el('orb').setAttribute('aria-disabled', 'false');
  el('title').textContent = '누르고 말해보세요';
  el('eyebrow').textContent = '목소리에 응답하는 순간';
  el('instruction').textContent = '말하는 동안 누르고, 끝나면 손을 떼세요';
}
function hold(event) {
  event.preventDefault();
  if (event.pointerId !== undefined) el('orb').setPointerCapture(event.pointerId);
  document.body.dataset.state = 'listening';
  el('title').textContent = '말씀해주세요';
  el('eyebrow').textContent = '누르고 있는 모습 미리보기';
  el('instruction').textContent = '실제 마이크는 켜지지 않습니다';
}
el('connection-label').textContent = '디자인 미리보기';
el('footer-copy').textContent = '화면 검토용 · 실제 녹음과 재생은 하지 않습니다';
el('orb').addEventListener('pointerdown', hold);
['pointerup', 'pointercancel', 'lostpointercapture'].forEach((name) => el('orb').addEventListener(name, ready));
el('orb').addEventListener('contextmenu', (event) => event.preventDefault());
el('orb').addEventListener('keydown', (event) => {if ([' ', 'Enter'].includes(event.key)) hold(event);});
el('orb').addEventListener('keyup', ready);
window.addEventListener('blur', ready);
el('connection').onclick = () => {el('help-status').textContent = '화면 검토용 파일입니다. 실제 연결은 웹실행.command로 시작하세요.';el('help').showModal();};
el('close-help').onclick = () => el('help').close();
el('reconnect').textContent = '닫기';
el('reconnect').onclick = () => el('help').close();
ready();
</script></body>''')
    target = HERE / '화면미리보기.html'
    target.write_text(page, encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
