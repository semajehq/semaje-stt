// TypeScript client for the Whisper STT service — lives beside the contract
// it implements (services/whisper/contract.md). The only way Node code
// talks to Whisper.
import type { WhisperResult } from '@semaje/schemas'

const WHISPER_URL = () => process.env.WHISPER_URL || 'http://localhost:8011'

export async function whisperHealth(): Promise<{ ok: boolean; backend?: string; model?: string }> {
  try {
    const res = await fetch(`${WHISPER_URL()}/health`, { signal: AbortSignal.timeout(2000) })
    if (!res.ok) return { ok: false }
    return await res.json() as never
  } catch {
    return { ok: false }
  }
}

async function whisperJson(path: string, init?: RequestInit): Promise<any> {
  const res = await fetch(`${WHISPER_URL()}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    signal: AbortSignal.timeout(30 * 60 * 1000),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || `Whisper service error ${res.status}`)
  return data
}

export async function whisperModels() {
  return whisperJson('/models')
}

export async function whisperDownloadModel(backend: string, model: string) {
  return whisperJson('/models/download', {
    method: 'POST', body: JSON.stringify({ backend, model }),
  })
}

export async function whisperActivateModel(backend: string, model: string) {
  return whisperJson('/models/activate', {
    method: 'POST', body: JSON.stringify({ backend, model }),
  })
}

export async function whisperDeleteModel(backend: string, model: string) {
  return whisperJson(`/models/${encodeURIComponent(backend)}/${encodeURIComponent(model)}`, {
    method: 'DELETE',
  })
}

export async function whisperTranscribe(
  audio: Buffer, filename: string, mimeType: string,
  opts: { language?: string; task?: 'transcribe' | 'translate' } = {},
): Promise<WhisperResult> {
  const form = new FormData()
  form.append('audio', new Blob([new Uint8Array(audio)], { type: mimeType || 'audio/mpeg' }), filename)
  if (opts.language) form.append('language', opts.language)
  form.append('task', opts.task || 'transcribe')

  let res: Response
  try {
    res = await fetch(`${WHISPER_URL()}/transcribe`, { method: 'POST', body: form })
  } catch (err: any) {
    throw new Error(`Whisper service unreachable at ${WHISPER_URL()} — is it running? (${err.message})`)
  }
  const data: any = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || `Whisper service error ${res.status}`)
  return data as WhisperResult
}
