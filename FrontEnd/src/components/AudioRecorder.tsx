import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { getBestRecorderMimeType } from "@/utils/audio"
import { Mic, Square } from "lucide-react"

interface AudioRecorderProps {
  onAudioReady: (file: File) => void
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onAudioReady }) => {
  const [isRecording, setIsRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)

  const mimeType = useMemo(() => getBestRecorderMimeType(), [])

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
      mediaRecorderRef.current?.stream.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const start = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream, { mimeType })
      chunksRef.current = []
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
      }
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType })
        const ext = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "m4a" : "webm"
        const file = new File([blob], `recording.${ext}`)
        onAudioReady(file)
        // Stop input tracks to release mic
        stream.getTracks().forEach((t) => t.stop())
      }
      mediaRecorderRef.current = mr
      mr.start()
      setIsRecording(true)
      setElapsed(0)
      timerRef.current = window.setInterval(() => setElapsed((s) => s + 1), 1000)
    } catch (e: any) {
      setError(e?.message || "Microphone access denied")
    }
  }, [mimeType, onAudioReady])

  const stop = useCallback(() => {
    if (!mediaRecorderRef.current) return
    mediaRecorderRef.current.stop()
    mediaRecorderRef.current = null
    setIsRecording(false)
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  return (
    <div className="flex flex-col gap-3 p-4 rounded-lg border bg-card">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant="secondary">Microphone</Badge>
          {isRecording ? (
            <span className="text-sm text-muted-foreground">Recording… {elapsed}s</span>
          ) : (
            <span className="text-sm text-muted-foreground">Ready</span>
          )}
        </div>
        {isRecording ? (
          <Button variant="destructive" onClick={stop} size="sm">
            <Square /> Stop
          </Button>
        ) : (
          <Button variant="hero" onClick={start} size="sm">
            <Mic /> Start
          </Button>
        )}
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      <p className="text-xs text-muted-foreground">Works in Chrome and Firefox. We will convert the recording to WAV on submit.</p>
    </div>
  )
}

export default AudioRecorder
