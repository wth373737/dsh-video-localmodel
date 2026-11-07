/**
 * Minimal Ollama client for the local screening pipeline.
 *
 * It talks the same HTTP API (and the same default base URL) as the installed
 * dsh-plugin-ollama-bridge, so both plugins can share one Ollama daemon and one
 * model inventory. Nothing is imported from DSH here on purpose: the plugin
 * stays loadable across harness upgrades.
 */

export const DEFAULT_BASE_URL = 'http://127.0.0.1:11434'

export function endpoint(baseURL, path) {
  return `${String(baseURL || DEFAULT_BASE_URL).replace(/\/+$/, '')}${path}`
}

export function describeFailure(error, baseURL) {
  const text = String(error?.cause?.code ?? error?.code ?? error?.message ?? error)
  if (/ECONNREFUSED|fetch failed|ENOTFOUND|EAI_AGAIN/i.test(text)) {
    return `无法连接 Ollama（${baseURL || DEFAULT_BASE_URL}）：${text}。请确认 ollama 服务在运行（终端执行 ollama list 能正常返回）。`
  }
  return `调用 Ollama 失败：${text}`
}

async function request(baseURL, path, options = {}) {
  const { body, timeoutMs = 300000, signal } = options
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error(`timeout ${timeoutMs}ms`)), timeoutMs)
  const relay = () => controller.abort(new Error('cancelled'))
  if (signal?.addEventListener) signal.addEventListener('abort', relay, { once: true })
  try {
    const response = await fetch(endpoint(baseURL, path), {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    })
    if (!response.ok) {
      const detail = await response.text().catch(() => '')
      throw new Error(`HTTP ${response.status}${detail ? `: ${detail.slice(0, 300)}` : ''}`)
    }
    return await response.json()
  } finally {
    clearTimeout(timer)
    if (signal?.removeEventListener) signal.removeEventListener('abort', relay)
  }
}

/** Installed models, exactly as `/api/tags` reports them. */
export async function listModels(baseURL, signal) {
  const payload = await request(baseURL, '/api/tags', { timeoutMs: 15000, signal })
  return Array.isArray(payload?.models) ? payload.models : []
}

/** One model's details, including the capabilities that decide its role. */
export async function showModel(baseURL, model, signal) {
  const payload = await request(baseURL, '/api/show', { body: { model }, timeoutMs: 20000, signal })
  return {
    model,
    family: String(payload?.details?.family ?? ''),
    parameters: String(payload?.details?.parameter_size ?? ''),
    capabilities: Array.isArray(payload?.capabilities) ? payload.capabilities.map(String) : [],
  }
}

/**
 * One chat completion. `images` are base64 JPEG/PNG payloads attached to the
 * single user message; `format: 'json'` asks the server for JSON output.
 */
export async function chat(baseURL, options = {}) {
  const {
    model,
    prompt,
    system,
    images,
    format,
    think = false,
    options: modelOptions = {},
    timeoutMs = 300000,
    signal,
    retries = 1,
  } = options
  const message = { role: 'user', content: String(prompt ?? '') }
  if (Array.isArray(images) && images.length > 0) message.images = images
  const messages = []
  if (system) messages.push({ role: 'system', content: String(system) })
  messages.push(message)
  const body = {
    model,
    stream: false,
    messages,
    options: {
      temperature: 0.1,
      num_ctx: 4096,
      num_predict: 600,
      ...modelOptions,
    },
    ...(format ? { format } : {}),
    ...(think === false ? { think: false } : {}),
  }
  let lastError
  for (let attempt = 1; attempt <= retries + 1; attempt++) {
    if (signal?.aborted) throw new Error('cancelled')
    const started = Date.now()
    try {
      const payload = await request(baseURL, '/api/chat', { body, timeoutMs, signal })
      if (payload?.error) throw new Error(String(payload.error))
      return {
        content: String(payload?.message?.content ?? ''),
        thinking: String(payload?.message?.thinking ?? ''),
        model: String(payload?.model ?? model),
        evalCount: Number(payload?.eval_count ?? 0),
        promptEvalCount: Number(payload?.prompt_eval_count ?? 0),
        evalDurationMs: Math.round(Number(payload?.eval_duration ?? 0) / 1e6),
        totalMs: Date.now() - started,
      }
    } catch (error) {
      lastError = error
      if (/cancelled/i.test(String(error?.message))) throw new Error('cancelled')
      if (attempt > retries) break
      await new Promise((done) => setTimeout(done, 800 * attempt))
    }
  }
  throw new Error(describeFailure(lastError, baseURL))
}

/** Remove ``` fences and pick the first balanced JSON object in a reply. */
export function parseJsonLoose(text) {
  const raw = String(text ?? '').trim()
  if (!raw) return undefined
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(raw)
  const candidate = fenced ? fenced[1].trim() : raw
  try {
    return JSON.parse(candidate)
  } catch {
    /* fall through to brace matching */
  }
  const start = candidate.indexOf('{')
  if (start < 0) return undefined
  let depth = 0
  let inString = false
  let escaped = false
  for (let index = start; index < candidate.length; index++) {
    const char = candidate[index]
    if (inString) {
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') inString = false
      continue
    }
    if (char === '"') inString = true
    else if (char === '{') depth++
    else if (char === '}') {
      depth--
      if (depth === 0) {
        try {
          return JSON.parse(candidate.slice(start, index + 1))
        } catch {
          return undefined
        }
      }
    }
  }
  return undefined
}

/**
 * Ask a vision model for one frame. Thinking models (qwen3-vl) spend tokens on
 * a reasoning block first, so an empty `content` is retried with a larger
 * budget instead of being reported as a model failure.
 */
export async function describeFrame(baseURL, options = {}) {
  const { model, image, at, prompt, signal, timeoutMs = 180000 } = options
  const ask = (numPredict, strict = false) =>
    chat(baseURL, {
      model,
      prompt: strict ? `${prompt}\n\n必须只输出 JSON，且 {"caption":"..."} 里的 caption 不能为空。` : prompt,
      images: [image],
      format: 'json',
      think: false,
      signal,
      timeoutMs,
      options: { num_predict: numPredict, num_ctx: 2048, temperature: 0.1 },
    })
  let answer = await ask(600)
  if (!answer.content.trim()) answer = await ask(1000)
  let parsed = parseJsonLoose(answer.content) ?? {}
  let caption = String(parsed.caption ?? parsed.description ?? '').trim()
  if (!caption) {
    // A well-formed JSON with an empty caption is a failed frame, not a valid
    // one: retry once with a stricter instruction, then let the caller record
    // the failure instead of storing a blank description.
    answer = await ask(900, true)
    parsed = parseJsonLoose(answer.content) ?? {}
    caption = String(parsed.caption ?? parsed.description ?? '').trim()
  }
  if (!caption) throw new Error('视觉模型没有给出有效描述（caption 为空，已重试一次）')
  return {
    at,
    caption,
    importance: clamp(Number(parsed.importance ?? parsed.score ?? 3), 1, 5),
    tags: Array.isArray(parsed.tags) ? parsed.tags.map((tag) => String(tag).slice(0, 20)).slice(0, 5) : [],
    raw: answer.content.trim(),
    ms: answer.totalMs,
    evalCount: answer.evalCount,
    hadThinking: Boolean(answer.thinking),
  }
}

export function clamp(value, low, high) {
  if (!Number.isFinite(value)) return low
  return Math.min(high, Math.max(low, value))
}
