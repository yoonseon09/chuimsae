// Only connected to a microphone while the button is held. Audio output stays silent.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Int16Array(1024);
    this.used = 0;
    this.finished = false;
    this.meterSamples = 0;
    this.port.onmessage = ({data}) => {
      if (data === 'flush') {
        this.finished = true;
        this.flush();
        this.port.postMessage({type: 'flushed'});
      }
    };
  }
  flush() {
    if (!this.used) return;
    const pcm = this.buffer.slice(0, this.used);
    this.port.postMessage({type: 'pcm', buffer: pcm.buffer}, [pcm.buffer]);
    this.used = 0;
  }
  process(inputs) {
    if (this.finished) return false;
    const input = inputs[0]?.[0];
    if (!input) return true;
    let energy = 0;
    for (const value of input) {
      const sample = Math.max(-1, Math.min(1, value));
      this.buffer[this.used++] = Math.round(sample * (sample < 0 ? 32768 : 32767));
      energy += sample * sample;
      if (this.used === this.buffer.length) this.flush();
    }
    this.meterSamples += input.length;
    if (this.meterSamples >= 1024) {
      this.port.postMessage({type: 'level', value: Math.sqrt(energy / input.length)});
      this.meterSamples = 0;
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
