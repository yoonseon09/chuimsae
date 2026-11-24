"""Build a self-contained design review file; it never accesses the microphone."""
from pathlib import Path
from urllib.parse import quote
import base64

HERE = Path(__file__).resolve().parent


def build():
    page = (HERE / 'web/index.html').read_text()
    style = (HERE / 'web/style.css').read_text()
    for font in (HERE / 'web/fonts').glob('*.woff2'):
        encoded = base64.b64encode(font.read_bytes()).decode('ascii')
        style = style.replace(f'/fonts/{font.name}', f'data:font/woff2;base64,{encoded}')
    for name in ('royal-landscape.png', 'pine-branches.png', 'responding-pine.png'):
        encoded = base64.b64encode((HERE / 'web/assets' / name).read_bytes()).decode('ascii')
        page = page.replace(f'/assets/{name}', f'data:image/png;base64,{encoded}')
    icon = quote((HERE / 'web/icon.svg').read_text(), safe='')
    page = page.replace('<link rel="stylesheet" href="/style.css">', f'<style>{style}</style>')
    page = page.replace('href="/icon.svg"', f'href="data:image/svg+xml,{icon}"')
    page = page.replace('<script type="module" src="/app.js"></script>', '')
    page = page.replace('href="/"', 'href="#"')
    page = page.replace('<title>추임새 — 목소리에 응답하는 순간</title>', '<title>추임새 · 디자인 미리보기</title>')
    page = page.replace('</body>', '''<script>
const el = (id) => document.getElementById(id);
let pointer = null;
function ready() {
  pointer = null;
  document.body.dataset.state = 'ready';
  document.body.dataset.held = 'false';
  el('orb').setAttribute('aria-disabled', 'false');
  el('orb').setAttribute('aria-pressed', 'false');
  el('title').textContent = '나무를 길게 누르고 추임새 얼쑤, 좋다, 잘한다를 말해보세요';
  el('instruction').textContent = '나무를 길게 누르고 말한 뒤 손을 떼세요';
}
function hold() {
  document.body.dataset.state = 'listening';
  document.body.dataset.held = 'true';
  el('orb').setAttribute('aria-pressed', 'true');
  el('title').textContent = '음성 인식중';
  el('instruction').textContent = '나무에서 손을 떼면 추임새가 출력돼요.';
}
el('footer-copy').textContent = '화면 검토용 · 실제 녹음과 재생은 하지 않습니다';
el('orb').addEventListener('pointerdown', (event) => {
  if (!event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault();
  pointer = event.pointerId;
  el('orb').setPointerCapture(pointer);
  hold();
});
el('orb').addEventListener('pointerup', (event) => {if (event.pointerId === pointer) ready();});
['pointercancel', 'lostpointercapture', 'blur'].forEach((name) => el('orb').addEventListener(name, ready));
el('orb').addEventListener('contextmenu', (event) => event.preventDefault());
el('orb').addEventListener('keydown', (event) => {
  if (![' ', 'Enter'].includes(event.key)) return;
  event.preventDefault();
  if (!event.repeat) hold();
});
el('orb').addEventListener('keyup', (event) => {if ([' ', 'Enter'].includes(event.key)) {event.preventDefault();ready();}});
window.addEventListener('blur', ready);
window.addEventListener('pagehide', ready);
document.addEventListener('visibilitychange', () => {if (document.hidden) ready();});
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
