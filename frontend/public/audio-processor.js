/**
 * audio-processor.js — Web Audio Worklet for capturing microphone input
 * 
 * Replaces deprecated ScriptProcessorNode with modern AudioWorkletProcessor
 * Captures 16kHz PCM-16 audio and sends to main thread via port.postMessage()
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // 1024 samples @ 16kHz = ~64ms chunks — low latency without gaps
    this.bufferSize = 1024;
    this.buffer = [];
    this.sampleCount = 0;
  }

  process(inputs, _outputs, _parameters) {
    const input = inputs[0]; // mono input
    if (input.length === 0) return true;

    const channelData = input[0]; // Get first (mono) channel

    // Convert Float32 to Int16 and accumulate
    for (let i = 0; i < channelData.length; i++) {
      const s = Math.max(-1, Math.min(1, channelData[i]));
      const pcm16 = s < 0 ? s * 0x8000 : s * 0x7FFF;
      this.buffer.push(pcm16);
      this.sampleCount++;
    }

    // Send chunks when buffer reaches bufferSize
    if (this.buffer.length >= this.bufferSize) {
      const int16Array = new Int16Array(this.buffer);
      this.port.postMessage({
        type: 'audio_chunk',
        data: int16Array.buffer,
        sampleCount: this.sampleCount
      }, [int16Array.buffer]); // Transfer ownership
      
      this.buffer = [];
    }

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);