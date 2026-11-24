const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');

function harness({pendingPermission = false} = {}) {
  const elements = new Map(), sent = [], timers = new Map(), listeners = {};
  let counter = 0, socket, node, allow;
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {dataset: {}, style: {setProperty() {}}, handlers: {},
      setAttribute(name, value) {this[name] = value;}, addEventListener(name, fn) {this.handlers[name] = fn;},
      setPointerCapture() {}, showModal() {}, close() {}});
    return elements.get(id);
  };
  const track = {stopped: false, stop() {this.stopped = true;}, addEventListener() {}};
  const stream = {getTracks: () => [track]};
  const graph = () => ({connect() {}, disconnect() {}});
  class Socket {
    static OPEN = 1;
    constructor() {socket = this; this.readyState = 0; this.bufferedAmount = 0;}
    send(value) {
      sent.push(value);
      if (typeof value !== 'string') return;
      const message = JSON.parse(value);
      if (message.type === 'begin') {
        this.onmessage({data: JSON.stringify({type: 'capture_ready', id: message.id})});
        this.onmessage({data: JSON.stringify({type: 'status', state: 'busy', phase: 'capturing'})});
      }
      if (message.type === 'clock') this.onmessage({data: JSON.stringify({type: 'clock', client_ms: message.client_ms, server_ms: 100000})});
    }
    close() {this.readyState = 3;}
  }
  class Audio {
    constructor() {this.state = 'suspended'; this.sampleRate = 48000; this.destination = {}; this.audioWorklet = {addModule: async () => {}};}
    async resume() {this.state = 'running';}
    async suspend() {this.state = 'suspended';}
    createMediaStreamSource() {return graph();}
    createGain() {return {...graph(), gain: {value: 1}};}
  }
  class Node {
    constructor() {
      node = this;
      this.port = {postMessage: () => queueMicrotask(() => this.port.onmessage({data: {type: 'flushed'}}))};
    }
    connect() {} disconnect() {}
  }
  const permission = pendingPermission ? new Promise((resolve) => {allow = () => resolve(stream);}) : Promise.resolve(stream);
  const context = vm.createContext({
    document: {getElementById: element, body: {dataset: {}}, documentElement: {style: {setProperty() {}}}, addEventListener(name, fn) {listeners[name] = fn;}},
    window: {isSecureContext: true, AudioWorkletNode: Node, addEventListener(name, fn) {listeners[name] = fn;}},
    navigator: {mediaDevices: {getUserMedia: () => permission}},
    location: {protocol: 'https:', host: 'localhost:8765', pathname: '/', search: '?key=secret'},
    history: {replaceState() {}}, sessionStorage: {setItem() {}, getItem: () => 'secret'},
    URLSearchParams, performance: {now: () => 1000}, crypto: {randomUUID: () => 'press'},
    WebSocket: Socket, AudioContext: Audio, AudioWorkletNode: Node, ArrayBuffer, Promise,
    setTimeout(fn) {timers.set(++counter, fn); return counter;}, clearTimeout(id) {timers.delete(id);},
    setInterval(fn) {timers.set(++counter, fn); return counter;}, clearInterval(id) {timers.delete(id);},
  });
  vm.runInContext(source + '\nglobalThis.api = {beginCapture, endCapture, cancelCapture, get active(){return active;}};', context);
  socket.readyState = 1; socket.onopen();
  socket.onmessage({data: JSON.stringify({type: 'status', state: 'ready', mode: 'preview'})});
  return {api: context.api, body: context.document.body, document: context.document, track, stream, sent, elements, listeners, timers, allow,
    get node() {return node;}, get socket() {return socket;}};
}

const settle = async () => {for (let i = 0; i < 12; i++) await Promise.resolve();};
const types = (app) => app.sent.map((m) => typeof m === 'string' ? JSON.parse(m).type : 'pcm');

test('release stops the actual microphone before flushing and sends PCM before end', async () => {
  const app = harness();
  await app.api.beginCapture();
  app.node.port.onmessage({data: {type: 'pcm', buffer: new ArrayBuffer(2048)}});
  app.api.endCapture();
  assert.equal(app.track.stopped, true);
  await settle();
  assert.equal(app.api.active, null);
  assert.deepEqual(types(app).slice(-3), ['begin', 'pcm', 'end']);
  assert.equal(JSON.parse(app.sent.at(-1)).samples, 1024);
});

test('releasing during a permission prompt cannot start a late recording', async () => {
  const app = harness({pendingPermission: true});
  const pending = app.api.beginCapture();
  app.api.endCapture();
  app.allow(); await pending;
  assert.equal(app.track.stopped, true);
  assert.equal(app.api.active, null);
  assert.equal(types(app).includes('begin'), false);
});

test('cancellation stops capture and never submits it for recognition', async () => {
  const app = harness(); await app.api.beginCapture();
  app.api.cancelCapture(); await settle();
  assert.equal(app.track.stopped, true);
  assert.equal(types(app).at(-1), 'cancel');
  assert.equal(types(app).includes('end'), false);
});

test('screen blur and server errors close the microphone', async () => {
  for (const reason of ['blur', 'server']) {
    const app = harness(); await app.api.beginCapture();
    if (reason === 'blur') app.listeners.blur();
    else app.socket.onmessage({data: JSON.stringify({type: 'error', detail: 'bad stream'})});
    assert.equal(app.track.stopped, true);
    assert.equal(app.api.active, null);
  }
});

test('network backlog cancels the press rather than submitting stale audio', async () => {
  const app = harness(); await app.api.beginCapture();
  app.socket.bufferedAmount = 300000;
  app.node.port.onmessage({data: {type: 'pcm', buffer: new ArrayBuffer(2048)}});
  assert.equal(app.track.stopped, true);
  assert.equal(types(app).includes('end'), false);
});

test('PCM worklet flushes the final partial chunk and excludes subsequent input', () => {
  let Processor;
  const chunks = [];
  const context = vm.createContext({AudioWorkletProcessor: class {constructor() {this.port = {postMessage: (data) => chunks.push(data)};}},
    registerProcessor: (_, cls) => {Processor = cls;}, Int16Array, Math});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/pcm-worklet.js'), 'utf8'), context);
  const processor = new Processor();
  processor.process([[Float32Array.from({length: 1100}, (_, i) => i % 2 ? 2 : -2)]]);
  processor.port.onmessage({data: 'flush'});
  const audio = chunks.filter((m) => m.type === 'pcm');
  assert.deepEqual(audio.map((m) => m.buffer.byteLength), [2048, 152]);
  assert.equal(new Int16Array(audio[0].buffer)[0], -32768);
  assert.equal(new Int16Array(audio[0].buffer)[1], 32767);
  assert.equal(chunks.at(-1).type, 'flushed');
  assert.equal(processor.process([[new Float32Array(128)]]), false);
});


test('press feedback begins during permission and ends immediately on release', async () => {
  const app = harness({pendingPermission: true});
  const pending = app.api.beginCapture();
  assert.equal(app.body.dataset.held, 'true');
  assert.equal(app.elements.get('orb')['aria-pressed'], 'true');
  app.api.endCapture();
  assert.equal(app.body.dataset.held, 'false');
  assert.equal(app.elements.get('orb')['aria-pressed'], 'false');
  app.allow(); await pending;
  assert.equal(app.body.dataset.held, 'false');
});

test('touch cancellation, lost pointer capture, focus loss and hidden page clear press feedback', async () => {
  for (const reason of ['pointercancel', 'lostpointercapture', 'blur', 'visibilitychange']) {
    const app = harness();
    const tree = app.elements.get('orb');
    tree.handlers.pointerdown({isPrimary: true, pointerType: 'touch', pointerId: 7, preventDefault() {}});
    await settle();
    if (reason === 'visibilitychange') {app.document.hidden = true; app.listeners.visibilitychange();}
    else tree.handlers[reason]();
    assert.equal(app.body.dataset.held, 'false', reason);
    assert.equal(app.track.stopped, true, reason);
    assert.equal(types(app).includes('end'), false, reason);
  }
});

test('space and Enter hold recording until key release', async () => {
  for (const key of [' ', 'Enter']) {
    const app = harness(), tree = app.elements.get('orb');
    const event = {key, repeat: false, preventDefault() {}};
    tree.handlers.keydown(event); await settle();
    assert.equal(app.body.dataset.held, 'true');
    tree.handlers.keydown({...event, repeat: true});
    assert.equal(types(app).filter((type) => type === 'begin').length, 1);
    tree.handlers.keyup(event);
    assert.equal(app.body.dataset.held, 'false');
    assert.equal(app.track.stopped, true);
    await settle();
    assert.equal(types(app).at(-1), 'end');
  }
});

test('recognized text stays above the tree through playback and server ready until the next press', async () => {
  const app = harness();
  const message = (msg) => app.socket.onmessage({data: JSON.stringify(msg)});
  message({type: 'recognized', text: '좋다, 잘한다!'});
  assert.equal(app.elements.get('title').textContent, '좋다, 잘한다!');
  assert.equal(app.body.dataset.state, 'complete');
  message({type: 'playback', command: '좋다', latency_ms: 230});
  assert.equal(app.elements.get('title').textContent, '좋다, 잘한다!');
  message({type: 'status', state: 'ready'});
  assert.equal(app.body.dataset.state, 'complete');
  assert.equal(app.elements.get('orb')['aria-disabled'], 'false');
  await app.api.beginCapture();
  assert.equal(app.body.dataset.state, 'listening');
  assert.equal(app.elements.get('title').textContent, '음성 인식중');
});

test('playback uses actual command if recognition is absent and server errors clear completion', () => {
  const app = harness();
  app.socket.onmessage({data: JSON.stringify({type: 'playback', command: '잘한다', latency_ms: 400})});
  assert.equal(app.elements.get('title').textContent, '잘한다');
  app.socket.onmessage({data: JSON.stringify({type: 'error', detail: '출력 오류'})});
  assert.equal(app.body.dataset.state, 'error');
  assert.equal(app.elements.get('result').dataset.kind, 'error');
  app.socket.onmessage({data: JSON.stringify({type: 'status', state: 'ready'})});
  assert.equal(app.body.dataset.state, 'ready');
});
