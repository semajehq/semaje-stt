// TypeScript client for the STT service — lives beside the contract it
// implements (services/stt/contract.md). The only way Node code talks to
// speech-to-text. Engines (Whisper, Omnilingual ASR) are chosen server-side.
import type { SttResult } from '@semaje/schemas'

const STT_URL = () => process.env.STT_URL || process.env.WHISPER_URL || 'http://localhost:8011'
const TRANSCRIBE_TIMEOUT_MS = () => Number(process.env.STT_TIMEOUT_MS || 6 * 60 * 60 * 1000)
const MAX_BUSY_RETRIES = 3

export class SttError extends Error {
  constructor(message: string, readonly status: number, readonly code: string, readonly requestId?: string) {
    super(message)
    this.name = 'SttError'
  }
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const token = process.env.STT_API_TOKEN
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...extra }
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

async function call(path: string, init: RequestInit & { timeoutMs?: number; requestId?: string } = {}): Promise<any> {
  const { timeoutMs = 30 * 60 * 1000, requestId, ...rest } = init
  for (let attempt = 0; ; attempt++) {
    let res: Response
    try {
      res = await fetch(`${STT_URL()}${path}`, {
        ...rest,
        headers: headers({ ...(requestId ? { 'x-request-id': requestId } : {}), ...(rest.headers as Record<string, string> || {}) }),
        signal: AbortSignal.timeout(timeoutMs),
      })
    } catch (err: any) {
      throw new SttError(`STT service unreachable at ${STT_URL()} — is it running? (${err.message})`, 503, 'unreachable')
    }
    const data: any = await res.json().catch(() => ({}))
    if (res.status === 503 && data.code === 'busy' && attempt < MAX_BUSY_RETRIES) {
      const wait = Math.min(Number(res.headers.get('retry-after') || 5), 60) * 1000
      await sleep(wait)
      continue
    }
    if (!res.ok) {
      throw new SttError(data.error || `STT service error ${res.status}`, res.status, data.code || 'stt_error', data.requestId)
    }
    return data
  }
}

export async function sttHealth(): Promise<{ ok: boolean; backend?: string; model?: string; engines?: Record<string, unknown> }> {
  try {
    return await call('/health', { timeoutMs: 2000 })
  } catch {
    return { ok: false }
  }
}

export async function sttLanguages(query: { q?: string; engine?: string } = {}) {
  const params = new URLSearchParams()
  if (query.q) params.set('q', query.q)
  if (query.engine) params.set('engine', query.engine)
  return call(`/languages${params.size ? `?${params}` : ''}`, { timeoutMs: 10_000 })
}

const json = (body: unknown): RequestInit => ({
  method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' },
})

export const sttModels = () => call('/models')
export const sttDownloadModel = (backend: string, model: string) => call('/models/download', json({ backend, model }))
export const sttActivateModel = (backend: string, model: string) => call('/models/activate', json({ backend, model }))
export const sttDeleteModel = (backend: string, model: string) =>
  call(`/models/${encodeURIComponent(backend)}/${encodeURIComponent(model)}`, { method: 'DELETE' })

export interface SttOptions {
  language?: string
  task?: 'transcribe' | 'translate'
  diarize?: boolean
  engine?: 'auto' | 'whisper' | 'omnilingual'
  channel?: string
  enhance?: boolean
  requestId?: string
}

export async function sttTranscribe(
  audio: Buffer, filename: string, mimeType: string, opts: SttOptions = {},
): Promise<SttResult> {
  const form = new FormData()
  form.append('audio', new Blob([new Uint8Array(audio)], { type: mimeType || 'application/octet-stream' }), filename)
  if (opts.language) form.append('language', opts.language)
  form.append('task', opts.task || 'transcribe')
  if (opts.diarize) form.append('diarize', 'true')
  if (opts.engine && opts.engine !== 'auto') form.append('engine', opts.engine)
  if (opts.channel) form.append('channel', opts.channel)
  if (opts.enhance) form.append('enhance', 'true')
  return call('/transcribe', { method: 'POST', body: form, timeoutMs: TRANSCRIBE_TIMEOUT_MS(), requestId: opts.requestId })
}
