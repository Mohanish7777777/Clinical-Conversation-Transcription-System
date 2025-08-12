import React, { useEffect, useMemo, useRef, useState } from "react"
import { Helmet } from "react-helmet-async"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import AudioRecorder from "@/components/AudioRecorder"
import { fileToWavFile } from "@/utils/audio"
import { useToast } from "@/hooks/use-toast"
import { useLocation } from "react-router-dom"
import Header from "@/components/Header"

interface ApiSegment {
  id: string
  startSec: number
  endSec: number
  speaker: string
  text: string
  lang: string
}

interface ApiSpeaker {
  id: string
  confidence?: number
  role?: string
}

interface ApiResponse {
  createdAt: string
  detectedLanguages: string[]
  encounterId: string
  segments: ApiSegment[]
  speakers?: ApiSpeaker[]
}

const Index: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<ApiResponse | null>(null)
  const { toast } = useToast()
  const location = useLocation()
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const canonicalUrl = useMemo(() => {
    const base = typeof window !== "undefined" ? window.location.origin : ""
    return `${base}${location.pathname}`
  }, [location.pathname])

  const onFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setResult(null)
    setSelectedFile(f)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(f))
  }

  const onRecorded = (file: File) => {
    setResult(null)
    setSelectedFile(file)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const ensureWav = async (file: File) => {
    const name = file.name?.toLowerCase() || "audio"
    const needsConvert = !name.endsWith(".wav") || !file.type.includes("wav")
    if (needsConvert || file.type.includes("mpeg") || name.endsWith(".mp3")) {
      return await fileToWavFile(file, name.replace(/\.[^.]+$/, "") + ".wav")
    }
    return file
  }

  const submit = async () => {
    if (!selectedFile) {
      toast({ title: "No audio selected", description: "Upload or record audio first.", variant: "default" as any })
      return
    }
    setSubmitting(true)
    setResult(null)
    try {
      const wavFile = await ensureWav(selectedFile)
      const fd = new FormData()
      fd.append("file", wavFile, wavFile.name || "audio.wav")
      const res = await fetch("http://127.0.0.1:5000/transcribe", {
        method: "POST",
        body: fd,
      })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data: ApiResponse = await res.json()
      setResult(data)
      toast({ title: "Transcription complete", description: `Detected ${data.detectedLanguages?.join(", ") || "—"}` })
    } catch (e: any) {
      console.error(e)
      toast({ title: "Upload failed", description: e?.message || "Unknown error", variant: "destructive" as any })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <>
      <Helmet>
        <title>healthpilot.ai — Audio Interpreter</title>
        <meta name="description" content="healthpilot.ai themed app: upload or record audio, auto-convert MP3→WAV, and transcribe with speaker timestamps." />
        <link rel="canonical" href={canonicalUrl} />
      </Helmet>

      <Header />

      <main className="container py-8 grid gap-6 md:grid-cols-2">
        <section>
          <Card className="relative overflow-hidden">
            <CardHeader>
              <CardTitle>Input</CardTitle>
              <CardDescription>Choose file or record. We’ll convert to WAV before sending.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium">Upload audio</label>
                <div className="flex items-center gap-3">
                  <Input ref={fileInputRef} type="file" accept="audio/wav,audio/x-wav,audio/mp3,audio/mpeg,audio/webm,audio/ogg" onChange={onFileChange} />
                  {selectedFile && (
                    <Badge variant="secondary">{selectedFile.name}</Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">Accepted: .wav, .mp3 (auto-convert to .wav), plus recorded formats.</p>
              </div>

              <div className="flex items-center">
                <div className="h-px flex-1 bg-border" />
                <span className="px-3 text-xs text-muted-foreground">or</span>
                <div className="h-px flex-1 bg-border" />
              </div>

              <AudioRecorder onAudioReady={onRecorded} />

              {previewUrl && (
                <div className="mt-2">
                  <audio src={previewUrl} controls className="w-full" />
                </div>
              )}

              <div className="flex items-center gap-3 pt-2">
                <Button variant="hero" onClick={submit} disabled={submitting || !selectedFile} aria-busy={submitting}>
                  {submitting && (
                    <span className="mr-2 inline-block size-4 animate-spin rounded-full border-2 border-ring border-t-transparent" aria-hidden="true" />
                  )}
                  {submitting ? "Submitting…" : "Submit for Transcription"}
                </Button>
                <Button variant="ghost" onClick={() => { setSelectedFile(null); setResult(null); if (previewUrl) { URL.revokeObjectURL(previewUrl); setPreviewUrl(null) } if (fileInputRef.current) fileInputRef.current.value = "" }}>
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>

        <section>
          <Card>
            <CardHeader>
              <CardTitle>Results</CardTitle>
              <CardDescription>Detected languages, speakers, and timecoded segments.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {result ? (
                <div className="space-y-4">
                  {result.detectedLanguages?.length ? (
                    <div className="flex flex-wrap gap-2">
                      <span className="text-sm text-muted-foreground">Languages:</span>
                      {result.detectedLanguages.map((l) => (
                        <Badge key={l} variant="secondary">{l}</Badge>
                      ))}
                    </div>
                  ) : null}

                  {result.speakers?.length ? (
                    <div className="flex flex-wrap gap-2">
                      <span className="text-sm text-muted-foreground">Speakers:</span>
                      {result.speakers.map((s) => (
                        <Badge key={s.id} variant="outline">
                          {s.id}{typeof s.confidence === "number" ? ` • conf ${(s.confidence * 100).toFixed(0)}%` : ""}
                        </Badge>
                      ))}
                    </div>
                  ) : null}

                  <div className="overflow-x-auto rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="whitespace-nowrap">Start (s)</TableHead>
                          <TableHead className="whitespace-nowrap">End (s)</TableHead>
                          <TableHead>Speaker</TableHead>
                          <TableHead>Text</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {result.segments?.length ? (
                          result.segments.map((seg) => (
                            <TableRow key={seg.id}>
                              <TableCell>{Number(seg.startSec).toFixed(2)}</TableCell>
                              <TableCell>{Number(seg.endSec).toFixed(2)}</TableCell>
                              <TableCell>
                                <Badge variant="secondary">{seg.speaker}</Badge>
                              </TableCell>
                              <TableCell className="max-w-[0] md:max-w-[480px] truncate" title={seg.text}>{seg.text || "—"}</TableCell>
                            </TableRow>
                          ))
                        ) : (
                          <TableRow>
                            <TableCell colSpan={4} className="text-center text-muted-foreground">No segments</TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">No results yet. Submit audio to see transcription.</div>
              )}
            </CardContent>
          </Card>
        </section>
      </main>
    </>
  )
}

export default Index
