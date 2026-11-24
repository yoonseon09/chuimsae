const $ = (id) => document.getElementById(id);
const orb = $('orb');
const params = new URLSearchParams(location.search);
if (params.has('key')) {
  sessionStorage.setItem('chuimsae-key', params.get('key'));
  history.replaceState(null, '', location.pathname);
}
const key = sessionStorage.getItem('chuimsae-key');
let socket, reconnectTimer, clockTimer, context, moduleReady;
let active = null, available = false, mode = 'live', serverState = 'connecting';
let clockOffset = null, clockRtt = Infinity;
let pressedPointer = null;
let lastError = '';
const clocks = new Map();

const messages = {
  connecting: ['목소리에 응답하는 순간', '잠시만 기다려주세요', 'Mac과 연결하고 있어요'],
  warming: ['목소리에 응답하는 순간', 'Mac을 준비하고 있어요', '처음에는 모델 준비에 시간이 걸릴 수 있어요'],
  ready: ['목소리에 응답하는 순간', '누르고 말해보세요', '말하는 동안 누르고, 끝나면 손을 떼세요'],
  permission: ['처음 한 번만', '마이크를 허용해주세요', '허용한 뒤 원을 다시 누르고 말해주세요'],
  starting: ['당신의 목소리를 기다려요', '마이크를 여는 중이에요', '원이 반응하면 누른 채 말씀하세요'],
  listening: ['지금, 듣고 있어요', '말씀해주세요', '끝나면 손을 떼세요'],
  processing: ['추임새를 찾고 있어요', '목소리를 전하고 있어요', 'Mac에서 인식하고 있습니다'],
  playing: ['목소리에 응답하는 순간', '추임새가 도착했어요', 'Mac 스피커에서 재생하고 있어요'],
  busy: ['다음 목소리를 기다려요', '잠시만 기다려주세요', 'Mac이 앞선 추임새를 처리하고 있어요'],
  error: ['연결을 확인해주세요', '잠시 연결이 멈췄어요', '오른쪽 위 연결 안내를 확인해주세요'],
};

function display(state, custom) {
  document.body.dataset.state = state;
  const [eyebrow, title, instruction] = custom || messages[state];
  $('eyebrow').textContent = eyebrow;
  $('title').textContent = title;
  $('instruction').textContent = instruction;
  orb.setAttribute('aria-disabled', String(!available || !['ready', 'listening'].includes(state)));
  orb.setAttribute('aria-pressed', String(state === 'listening'));
}
function result(title, detail = '', time = '', kind = 'normal') {
  $('result').hidden = false;
  $('result').dataset.kind = kind;
  $('result-title').textContent = title;
  $('result-detail').textContent = detail;
  $('result-time').textContent = time;
}
function error(message) {
  lastError = message;
  $('help-status').textContent = message;
  result(message, '원을 다시 누르거나 연결 안내를 확인해주세요', '', 'error');
}
function applyServerState(data) {
  if (data.state === 'error' && active) cancelCapture();
  serverState = data.state;
  mode = data.mode || mode;
  available = data.state === 'ready';
  $('connection-label').textContent = mode === 'preview' ? '디자인 미리보기' : ({ready: 'Mac 연결됨', warming: 'Mac 준비 중', error: 'Mac 확인 필요', busy: 'Mac 연결됨'}[data.state] || 'Mac 연결 중');
  $('help-status').textContent = data.detail || (available ? 'Mac과 연결되어 있어요. 소리는 Mac에서 재생됩니다.' : 'Mac의 준비 상태를 확인해주세요.');
  if (data.state === 'error') lastError = data.detail;
  if (!active) display(data.state === 'busy' && data.phase === 'processing' ? 'processing' : data.state);
  if (mode === 'preview') $('footer-copy').textContent = '디자인 미리보기 · 음성 인식과 스피커 출력은 연결 전';
}
function send(data) {
  if (socket?.readyState !== WebSocket.OPEN) throw new Error('Mac과 연결이 끊겼어요.');
  socket.send(typeof data === 'string' || data instanceof ArrayBuffer ? data : JSON.stringify(data));
}
function syncClock() {
  if (socket?.readyState !== WebSocket.OPEN) return;
  const stamp = performance.now();
  clocks.set(stamp, stamp);
  if (clocks.size > 10) clocks.delete(clocks.keys().next().value);
  send({type: 'clock', client_ms: stamp});
}
function connect() {
  clearTimeout(reconnectTimer);
  clearInterval(clockTimer);
  clockTimer = null;
  if (socket) { socket.onclose = null; socket.close(); }
  available = false;
  display('connecting');
  if (!window.isSecureContext || !navigator.mediaDevices || !window.AudioWorkletNode) {
    display('error');
    error('마이크를 사용하려면 신뢰된 HTTPS 주소로 접속해주세요.');
    $('connection-label').textContent = 'HTTPS 설정 필요';
    return;
  }
  if (!key) {
    display('error');
    error('Mac 터미널에 나온 휴대폰 접속 주소를 열어주세요.');
    $('connection-label').textContent = '연결 주소 확인';
    return;
  }
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.onopen = () => send({type: 'auth', key});
  socket.onmessage = ({data}) => {
    const msg = JSON.parse(data);
    if (msg.type === 'status') {
      applyServerState(msg);
      if (!clockTimer) {
        syncClock();
        clockTimer = setInterval(syncClock, 5000);
      }
    } else if (msg.type === 'clock') {
      const sent = clocks.get(msg.client_ms);
      if (sent === undefined) return;
      clocks.delete(msg.client_ms);
      const now = performance.now(), rtt = now - sent;
      if (rtt < clockRtt || clockRtt === Infinity) {
        clockRtt = rtt;
        clockOffset = msg.server_ms - (sent + now) / 2;
      }
      $('diagnostic').textContent = `연결 확인 왕복 시간: ${Math.round(rtt)} ms`;
    } else if (msg.type === 'capture_ready') {
      if (active?.id === msg.id) active.accept?.();
    } else if (msg.type === 'rejected') {
      active?.reject?.(new Error(msg.detail));
      error(msg.detail);
    } else if (msg.type === 'recognized') {
      result(`“${msg.text}”`, msg.detail || 'Mac 스피커로 보내고 있어요');
    } else if (msg.type === 'playback') {
      display('playing');
      const timingDetail = msg.timing_synced === false ? 'Mac 수신 기준 추정 · 전송 지연 제외' : 'Mac 스피커 출력 예정 · 말 끝 기준 추정';
      result(`“${msg.command}”`, mode === 'preview' ? '미리보기에서는 재생하지 않아요' : timingDetail, `${(msg.latency_ms / 1000).toFixed(2)}초`);
    } else if (msg.type === 'outcome') {
      if (msg.kind !== 'played') result(msg.title, msg.detail, '', msg.kind === 'error' ? 'error' : 'normal');
    } else if (msg.type === 'error') {
      if (active) cancelCapture();
      error(msg.detail);
    }
  };
  socket.onerror = () => { $('help-status').textContent = 'Mac 실행 상태, Wi-Fi, HTTPS 인증서를 확인해주세요.'; };
  socket.onclose = (event) => {
    available = false;
    cancelCapture();
    clearInterval(clockTimer); clockTimer = null;
    clockOffset = null; clockRtt = Infinity;
    display('error');
    $('connection-label').textContent = 'Mac 연결 끊김';
    if (event.code === 1008) {
      error('접속 주소가 만료됐어요. Mac 터미널의 새 주소를 열어주세요.');
    } else reconnectTimer = setTimeout(connect, 2500);
  };
}

function stopTracks(session) {
  session.stream?.getTracks().forEach((track) => track.stop());
}
function disconnectAudio(session) {
  clearTimeout(session.limitTimer); clearTimeout(session.flushTimer);
  session.source?.disconnect(); session.node?.disconnect(); session.mute?.disconnect();
  if (context?.state === 'running') context.suspend().catch(() => {});
  document.documentElement.style.setProperty('--level', '0');
}
function cancelCapture() {
  const session = active;
  if (!session) return;
  active = null;
  session.held = false; session.cancelled = true;
  stopTracks(session); disconnectAudio(session);
  session.reject?.(new Error('입력이 취소됐어요.'));
  if (socket?.readyState === WebSocket.OPEN) send({type: 'cancel', id: session.id});
  if (available) display('ready');
}
async function beginCapture() {
  if (active || !available || !key) return;
  const session = {id: crypto.randomUUID(), held: true, stage: 'starting', samples: 0};
  active = session;
  $('result').hidden = true;
  display('starting');
  try {
    // Resume from the pointer/key gesture, before awaiting permission (Safari).
    if (!context || context.state === 'closed') {
      context = new AudioContext({latencyHint: 'interactive'});
      moduleReady = context.audioWorklet.addModule('/pcm-worklet.js');
    }
    const resumed = context.resume();
    const stream = await navigator.mediaDevices.getUserMedia({audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    }, video: false});
    session.stream = stream;
    stream.getTracks().forEach((track) => track.addEventListener('ended', () => {
      if (active === session && session.held) {
        cancelCapture(); error('마이크 연결이 종료됐어요. 다시 눌러주세요.');
      }
    }));
    if (active !== session || !session.held) { stopTracks(session); return; }
    await Promise.all([moduleReady, resumed]);
    if (active !== session || !session.held) { stopTracks(session); return; }
    session.ready = new Promise((resolve, reject) => {session.accept = resolve; session.reject = reject;});
    send({type: 'begin', id: session.id, sample_rate: context.sampleRate});
    const ackTimer = setTimeout(() => session.reject(new Error('Mac의 입력 준비가 늦어지고 있어요.')), 2000);
    try { await session.ready; } finally { clearTimeout(ackTimer); }
    if (active !== session || !session.held) { stopTracks(session); return; }
    session.source = context.createMediaStreamSource(stream);
    session.node = new AudioWorkletNode(context, 'pcm-capture', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]});
    session.mute = context.createGain(); session.mute.gain.value = 0;
    session.node.port.onmessage = ({data}) => {
      if (active !== session || session.cancelled) return;
      if (data.type === 'pcm') {
        if (socket?.readyState !== WebSocket.OPEN || socket.bufferedAmount > 256 * 1024) {
          cancelCapture(); error('음성 전송이 밀리고 있어요. Wi-Fi 연결을 확인해주세요.'); return;
        }
        session.samples += data.buffer.byteLength / 2;
        send(data.buffer);
      } else if (data.type === 'level' && session.held) {
        document.documentElement.style.setProperty('--level', Math.min(1, data.value * 7).toFixed(3));
      } else if (data.type === 'flushed' && session.stage === 'ending') {
        finishCapture(session);
      }
    };
    session.source.connect(session.node); session.node.connect(session.mute); session.mute.connect(context.destination);
    session.stage = 'recording';
    display('listening');
    session.limitTimer = setTimeout(() => {
      endCapture(); result('입력을 마쳤어요', '한 번에 최대 10초까지 말할 수 있어요');
    }, 10000);
  } catch (err) {
    if (active !== session) { stopTracks(session); return; }
    cancelCapture();
    if (err.name === 'NotAllowedError') {
      display('permission'); error('브라우저 설정에서 마이크 사용을 허용해주세요.');
    } else error(err.message || '마이크를 열지 못했어요.');
  }
}
function endCapture() {
  const session = active;
  if (!session || !session.held) return;
  session.held = false;
  session.releasedAt = performance.now();
  // Stop the physical microphone immediately, even while permission/ack is pending.
  stopTracks(session);
  if (session.stage !== 'recording') { cancelCapture(); return; }
  session.stage = 'ending';
  clearTimeout(session.limitTimer);
  session.source.disconnect();
  display('processing');
  session.node.port.postMessage('flush');
  session.flushTimer = setTimeout(() => {
    if (active === session) { cancelCapture(); error('녹음 마무리가 지연돼 입력을 취소했어요. 다시 눌러주세요.'); }
  }, 700);
}
function finishCapture(session) {
  disconnectAudio(session);
  active = null;
  available = false;
  const timing = clockOffset === null ? {} : {
    released_server_ms: session.releasedAt + clockOffset,
    clock_uncertainty_ms: clockRtt / 2,
  };
  send({type: 'end', id: session.id, samples: session.samples, ...timing});
}

orb.addEventListener('pointerdown', (event) => {
  if (!event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault();
  if (!available) { if (serverState === 'error') $('help').showModal(); return; }
  pressedPointer = event.pointerId;
  orb.setPointerCapture(event.pointerId);
  beginCapture();
});
orb.addEventListener('pointerup', (event) => {
  if (event.pointerId !== pressedPointer) return;
  pressedPointer = null; endCapture();
});
orb.addEventListener('pointercancel', () => {pressedPointer = null; cancelCapture();});
orb.addEventListener('lostpointercapture', () => {if (pressedPointer !== null) {pressedPointer = null; cancelCapture();}});
orb.addEventListener('contextmenu', (event) => event.preventDefault());
orb.addEventListener('keydown', (event) => {
  if (![' ', 'Enter'].includes(event.key)) return;
  event.preventDefault(); if (!event.repeat) beginCapture();
});
orb.addEventListener('keyup', (event) => {if ([' ', 'Enter'].includes(event.key)) {event.preventDefault(); endCapture();}});
window.addEventListener('blur', cancelCapture);
window.addEventListener('pagehide', () => {cancelCapture(); socket?.close();});
document.addEventListener('visibilitychange', () => {if (document.hidden) cancelCapture();});
$('connection').onclick = () => {if (lastError) $('help-status').textContent = lastError; $('help').showModal();};
$('close-help').onclick = () => $('help').close();
$('reconnect').onclick = () => {cancelCapture(); $('help').close(); clockTimer = null; connect();};
connect();
