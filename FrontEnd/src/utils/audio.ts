export async function decodeToAudioBuffer(file: Blob): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer()
  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
  // On some browsers AudioContext starts suspended by default
  if (audioCtx.state === "suspended") {
    try {
      await audioCtx.resume()
    } catch {
      // ignore
    }
  }
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0) as ArrayBuffer)
  // Close the context to free resources
  audioCtx.close()
  return audioBuffer
}

function floatTo16BitPCM(output: DataView, offset: number, input: Float32Array) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]))
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
}

function writeWavHeader(view: DataView, sampleRate: number, numSamples: number, numChannels: number) {
  const bytesPerSample = 2
  const blockAlign = numChannels * bytesPerSample
  const byteRate = sampleRate * blockAlign
  const dataSize = numSamples * bytesPerSample * numChannels

  // RIFF identifier 'RIFF'
  view.setUint32(0, 0x52494646, false)
  // file length minus RIFF identifier length and file description length = 36 + dataSize
  view.setUint32(4, 36 + dataSize, true)
  // RIFF type 'WAVE'
  view.setUint32(8, 0x57415645, false)
  // format chunk identifier 'fmt '
  view.setUint32(12, 0x666d7420, false)
  // format chunk length 16
  view.setUint32(16, 16, true)
  // sample format (raw)
  view.setUint16(20, 1, true)
  // channel count
  view.setUint16(22, numChannels, true)
  // sample rate
  view.setUint32(24, sampleRate, true)
  // byte rate (sample rate * block align)
  view.setUint32(28, byteRate, true)
  // block align (channel count * bytes per sample)
  view.setUint16(32, blockAlign, true)
  // bits per sample
  view.setUint16(34, 16, true)
  // data chunk identifier 'data'
  view.setUint32(36, 0x64617461, false)
  // data chunk length
  view.setUint32(40, dataSize, true)
}

export function audioBufferToWavBlob(buffer: AudioBuffer, opts?: { forceMono?: boolean }): Blob {
  const numChannels = opts?.forceMono ? 1 : buffer.numberOfChannels
  const sampleRate = buffer.sampleRate

  // Downmix to mono if requested
  let channelData: Float32Array[] = []
  if (numChannels === 1) {
    channelData = [buffer.getChannelData(0)]
  } else {
    // Average channels to mono
    const length = buffer.length
    const tmp = new Float32Array(length)
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const data = buffer.getChannelData(c)
      for (let i = 0; i < length; i++) {
        tmp[i] += data[i] / buffer.numberOfChannels
      }
    }
    channelData = [tmp]
  }

  const numSamples = channelData[0].length
  const headerSize = 44
  const bytesPerSample = 2
  const dataSize = numSamples * bytesPerSample * numChannels
  const bufferSize = headerSize + dataSize

  const ab = new ArrayBuffer(bufferSize)
  const view = new DataView(ab)

  writeWavHeader(view, sampleRate, numSamples, numChannels)

  let offset = headerSize
  if (numChannels === 1) {
    floatTo16BitPCM(view, offset, channelData[0])
  } else {
    // Interleave channels (not used when forceMono true)
    const interleaved = new Float32Array(numSamples * numChannels)
    for (let i = 0; i < numSamples; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        interleaved[i * numChannels + ch] = buffer.getChannelData(ch)[i]
      }
    }
    floatTo16BitPCM(view, offset, interleaved)
  }

  return new Blob([view], { type: "audio/wav" })
}

export async function fileToWavFile(input: Blob, desiredName = "audio.wav", forceMono = true): Promise<File> {
  // If already WAV, just ensure File type
  // Note: some browsers report wav as 'audio/wav' or 'audio/x-wav'
  const isWav = (input.type && input.type.includes("wav")) || (input instanceof File && /\.wav$/i.test(input.name))
  if (isWav) {
    const file = input instanceof File ? input : new File([input], desiredName, { type: "audio/wav" })
    return file
  }
  const audioBuffer = await decodeToAudioBuffer(input)
  const wavBlob = audioBufferToWavBlob(audioBuffer, { forceMono })
  return new File([wavBlob], desiredName, { type: "audio/wav" })
}

export function getBestRecorderMimeType(): string {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ]
  for (const t of candidates) {
    // @ts-ignore - MediaRecorder may not exist on server
    if (typeof MediaRecorder !== "undefined" && (MediaRecorder as any).isTypeSupported?.(t)) {
      return t
    }
  }
  // Fallback
  return "audio/webm"
}
