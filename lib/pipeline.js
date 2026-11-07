/**
 * Pipeline orchestration for dsh-video-localmodel.
 *
 * Stage order is deliberate: deterministic media signals first, then local ASR,
 * then local VLM captions, then a local text model for the timeline. Every
 * stage caches into `<workspace>/.video/<key>/`, so a rerun resumes instead of
 * paying the same minutes twice.
 */
import { createHash } from 'node:crypto'
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { basename, dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  extractAudio,
  extractFrameAt,
  extractIntervalFrames,
  detectScenes,
  formatClock,
  formatTime,
  probe,
  run,
  which,
  FFMPEG_CANDIDATES,
  FFPROBE_CANDIDATES,
} from './media.js'
import { chat, describeFrame, listModels, parseJsonLoose, showModel } from './ollama.js'

export const DEFAULTS = {
  ffmpegPath: '',
  ffprobePath: '',
  pythonPath: '',
  asrModelPath: '',
  asrLanguage: 'Chinese',
  asrThreads: 0,
  asrBatch: 4,
  asrTargetSec: 25,
  asrMinSec: 12,
  ollamaBaseURL: 'http://127.0.0.1:11434',
  visionModel: '',
  textModel: '',
  coarseSec: 5,
  coarseWidth: 512,
  sceneThreshold: 0.28,
  maxFrames: 400,
  maxVideoMinutes: 30,
  captionPerCall: 60,
  keyFrameWidth: 1280,
  keepFrames: 12,
  textWindowChars: 1200,
  framesPerSegment: 2,
}

const VISION_PREFERENCE = [/qwen3?-?vl/i, /minicpm-?v/i, /llava/i, /gemma3/i, /moondream/i]
const TEXT_PREFERENCE = [/qwen2\.5:7b/i, /qwen2\.5:14b/i, /qwen3:8b/i, /deepseek-r1/i]

export function resolveSettings(root, config = {}) {
  const settings = { ...DEFAULTS }
  const layers = []
  const settingsPath = join(root, '.video', 'settings.json')
  const homeSettings = process.env.DSH_HOME ? join(process.env.DSH_HOME, 'video-localmodel.json') : ''
  for (const path of [settingsPath, homeSettings]) {
    if (!path || !existsSync(path)) continue
    try {
      const parsed = JSON.parse(readFileSync(path, 'utf8'))
      if (parsed && typeof parsed === 'object') {
        Object.assign(settings, normalizeSettings(parsed))
        layers.push(path)
      }
    } catch {
      /* an unreadable settings file must not break the tools */
    }
  }
  if (config && typeof config === 'object') {
    Object.assign(settings, normalizeSettings(config))
    layers.push('plugin config')
  }
  for (const [key, envName] of [
    ['ffmpegPath', 'DSH_VIDEO_FFMPEG'],
    ['ffprobePath', 'DSH_VIDEO_FFPROBE'],
    ['pythonPath', 'DSH_VIDEO_PYTHON'],
    ['asrModelPath', 'DSH_VIDEO_ASR_MODEL'],
    ['ollamaBaseURL', 'DSH_VIDEO_OLLAMA'],
    ['visionModel', 'DSH_VIDEO_VISION_MODEL'],
    ['textModel', 'DSH_VIDEO_TEXT_MODEL'],
  ]) {
    const value = process.env[envName]
    if (value) {
      settings[key] = value
      layers.push(envName)
    }
  }
  return { settings, source: layers }
}

function normalizeSettings(input) {
  const out = {}
  for (const key of Object.keys(DEFAULTS)) {
    if (input[key] === undefined || input[key] === null || input[key] === '') continue
    out[key] = typeof DEFAULTS[key] === 'number' ? Number(input[key]) : input[key]
  }
  return out
}

/* --------------------------------------------------------------- detection */

const envCache = { at: 0, value: undefined }

function pythonCandidates(settings) {
  const list = []
  const push = (path) => {
    if (path && !list.includes(path)) list.push(path)
  }
  push(settings.pythonPath)
  push(process.env.DSH_VIDEO_PYTHON)
  if (process.env.CONDA_PREFIX) push(join(process.env.CONDA_PREFIX, process.platform === 'win32' ? 'python.exe' : 'bin/python'))
  if (process.env.CONDA_EXE) {
    const root = dirname(dirname(process.env.CONDA_EXE))
    push(join(root, process.platform === 'win32' ? 'python.exe' : 'bin/python'))
  }
  const conda = which('conda', [])
  if (conda) {
    const root = dirname(dirname(conda))
    push(join(root, process.platform === 'win32' ? 'python.exe' : 'bin/python'))
  }
  for (const base of ['anaconda3', 'miniconda3', 'miniforge3', 'mambaforge']) {
    push(join(process.env.USERPROFILE ?? '', base, process.platform === 'win32' ? 'python.exe' : 'bin/python'))
    push(join(process.env.LOCALAPPDATA ?? '', base, process.platform === 'win32' ? 'python.exe' : 'bin/python'))
    push(join('C:\\ProgramData', base, process.platform === 'win32' ? 'python.exe' : 'bin/python'))
  }
  push(which('python', []))
  push(which('python3', []))
  return [...new Set(list.filter(Boolean))]
}

function modelCandidates(settings, roots) {
  const list = []
  const push = (path) => {
    if (path && !list.includes(path)) list.push(path)
  }
  push(settings.asrModelPath)
  push(process.env.DSH_VIDEO_ASR_MODEL)
  for (const root of roots) {
    if (!root || !existsSync(root)) continue
    let entries = []
    try {
      entries = readdirSync(root, { withFileTypes: true })
    } catch {
      continue
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue
      if (!/qwen3?[-_]?asr/i.test(entry.name)) continue
      const candidate = join(root, entry.name)
      if (existsSync(join(candidate, 'model.safetensors')) || existsSync(join(candidate, 'model.safetensors.index.json'))) {
        push(candidate)
      }
    }
  }
  return list.filter((path) => existsSync(path))
}

function hasModelWeights(directory) {
  if (!existsSync(directory)) return false
  if (!existsSync(join(directory, 'config.json'))) return false
  return (
    existsSync(join(directory, 'model.safetensors')) ||
    existsSync(join(directory, 'model.safetensors.index.json')) ||
    existsSync(join(directory, 'model.bin'))
  )
}

/** Verify that one python really can import the ASR stack, once per session. */
async function verifyPython(python, workdir, signal) {
  const tmp = join(workdir, '.video', 'tmp')
  mkdirSync(tmp, { recursive: true })
  try {
    const { stdout } = await run(
      python,
      ['-c', "import sys, torch, qwen_asr; print('OK', sys.version.split()[0], torch.__version__)"],
      {
        cwd: workdir,
        signal,
        timeoutMs: 120000,
        maxOutput: 64 * 1024,
        env: {
          ...process.env,
          TEMP: tmp,
          TMP: tmp,
          NUMBA_CACHE_DIR: join(tmp, 'numba'),
          PYTHONIOENCODING: 'utf-8',
        },
      },
    )
    const line = stdout.trim().split(/\r?\n/).filter(Boolean).pop() ?? ''
    return { ok: line.startsWith('OK'), detail: line }
  } catch (error) {
    return { ok: false, detail: String(error?.message ?? error).slice(0, 300) }
  }
}

export async function detectEnvironment(root, settings, options = {}) {
  const { signal, refresh = false } = options
  if (!refresh && envCache.value && Date.now() - envCache.at < 10 * 60 * 1000) return envCache.value

  const ffmpeg = settings.ffmpegPath && existsSync(settings.ffmpegPath)
    ? settings.ffmpegPath
    : which('ffmpeg', FFMPEG_CANDIDATES)
  const ffprobe = settings.ffprobePath && existsSync(settings.ffprobePath)
    ? settings.ffprobePath
    : which('ffprobe', FFPROBE_CANDIDATES)

  const modelRoots = []
  const seedRoots = [
    process.env.DSH_VIDEO_MODEL_ROOT,
    'D:\\AI_Models',
    join(process.env.USERPROFILE ?? '', 'AI_Models'),
    join(root, 'models'),
  ]
  for (const seed of seedRoots) {
    if (seed) modelRoots.push(seed)
  }

  const pythonAttempts = []
  let python
  let pythonDetail = ''
  for (const candidate of pythonCandidates(settings)) {
    const verdict = await verifyPython(candidate, root, signal)
    pythonAttempts.push({ path: candidate, ok: verdict.ok, detail: verdict.detail })
    if (verdict.ok) {
      python = candidate
      pythonDetail = verdict.detail
      break
    }
  }

  let asrModel = ''
  let asrDetail = ''
  for (const candidate of modelCandidates(settings, modelRoots)) {
    if (hasModelWeights(candidate)) {
      asrModel = candidate
      asrDetail = 'weights found'
      break
    }
  }

  let ollama = { reachable: false, error: '', models: [] }
  try {
    const models = await listModels(settings.ollamaBaseURL, signal)
    ollama = {
      reachable: true,
      error: '',
      models: models.map((entry) => ({
        name: String(entry?.name ?? entry?.model ?? ''),
        family: String(entry?.details?.family ?? ''),
        parameters: String(entry?.details?.parameter_size ?? ''),
        sizeBytes: Number(entry?.size ?? 0),
        capabilities: Array.isArray(entry?.capabilities) ? entry.capabilities.map(String) : [],
      })),
    }
  } catch (error) {
    ollama.error = String(error?.message ?? error)
  }

  const visionModels = ollama.models
    .filter((model) => model.capabilities.includes('vision') || /vl|vision|llava/i.test(model.name))
    .sort((a, b) => scorePreference(a.name, VISION_PREFERENCE) - scorePreference(b.name, VISION_PREFERENCE))
  const textModels = ollama.models
    .filter((model) => model.capabilities.includes('completion') || model.capabilities.length === 0)
    .sort((a, b) => scorePreference(a.name, TEXT_PREFERENCE) - scorePreference(b.name, TEXT_PREFERENCE))

  const visionModel = settings.visionModel || visionModels[0]?.name || ''
  const textModel = settings.textModel || textModels.find((model) => model.name === 'qwen2.5:7b')?.name || textModels[0]?.name || ''

  const value = {
    ffmpeg,
    ffprobe,
    python,
    pythonDetail,
    pythonAttempts,
    asrModel,
    asrDetail,
    ollama,
    visionModel,
    textModel,
    visionCapabilities: visionModels.map((model) => model.name),
    textCapabilities: textModels.map((model) => model.name),
    detectedAt: new Date().toISOString(),
  }
  envCache.value = value
  envCache.at = Date.now()
  return value
}

function scorePreference(name, preferences) {
  const index = preferences.findIndex((pattern) => pattern.test(name))
  return index < 0 ? 999 : index
}

/* ------------------------------------------------------------------ state */

export function videoKey(file) {
  const stat = statSync(file)
  const digest = createHash('sha1')
    .update(`${resolve(file)}|${stat.size}|${Math.round(stat.mtimeMs)}`)
    .digest('hex')
  return digest.slice(0, 16)
}

export function stateDir(root, file) {
  return join(root, '.video', videoKey(file))
}

export function readJson(path, fallback) {
  try {
    return JSON.parse(readFileSync(path, 'utf8'))
  } catch {
    return fallback
  }
}

export function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, `${JSON.stringify(value, null, 1)}\n`, 'utf8')
  return path
}

export function resolveInput(root, input) {
  const raw = String(input ?? '').trim()
  if (!raw) throw new Error('缺少 input：请传入视频文件的绝对路径（或相对会话工作区的路径）。')
  const target = isAbsolute(raw) ? raw : resolve(root, raw)
  if (!existsSync(target)) throw new Error(`找不到视频文件：${target}`)
  const stat = statSync(target)
  if (!stat.isFile()) throw new Error(`不是文件：${target}`)
  return { path: target, size: stat.size, mtimeMs: Math.round(stat.mtimeMs) }
}

/* ------------------------------------------------------------------- plan */

function estimate(duration, frameCount, mode, settings) {
  const audio = Math.max(0, duration)
  const asrSeconds = Math.round(audio / 2.4 + 3)
  const sceneSeconds = Math.round(audio * 0.03 + 2)
  const frameSeconds = Math.round(frameCount * 0.01 + 2)
  const captionSeconds = mode === 'fast' ? 0 : Math.round(frameCount * 4.2)
  const timelineSeconds = Math.round((audio / 180 + 1) * 5 + 5)
  const exportSeconds = 3
  return {
    scanSeconds: asrSeconds + sceneSeconds + frameSeconds,
    captionSeconds,
    timelineSeconds,
    exportSeconds,
    totalSeconds: asrSeconds + sceneSeconds + frameSeconds + captionSeconds + timelineSeconds + exportSeconds,
  }
}

export async function planFor(root, inputArg, args, settings, env) {
  const input = resolveInput(root, inputArg)
  const mode = args.mode === 'fast' ? 'fast' : 'standard'
  const probeInfo = await probe(env.ffprobe, input.path, { cwd: root })
  const duration = probeInfo.duration
  const maxMinutes = Number(args.max_minutes ?? settings.maxVideoMinutes)
  if (duration > maxMinutes * 60) {
    throw new Error(
      `视频长 ${Math.round(duration / 60)} 分钟，超过当前上限 ${maxMinutes} 分钟。提高 max_minutes 或先裁剪片段。`,
    )
  }
  const coarseSec = Math.max(1, Number(args.coarse_sec ?? settings.coarseSec))
  const frameCount = Math.min(settings.maxFrames, Math.ceil(duration / coarseSec))
  const visionModel = String(args.vision_model ?? env.visionModel ?? '')
  const textModel = String(args.text_model ?? env.textModel ?? '')
  const warnings = []
  if (!env.ffmpeg || !env.ffprobe) warnings.push('没有找到 ffmpeg/ffprobe：抽帧、抽音频和场景检测都无法进行。')
  if (!env.python) warnings.push(`没有找到能 import torch/qwen_asr 的 Python：ASR 不可用。${env.pythonAttempts?.[0]?.detail ? `最近一次输出：${env.pythonAttempts[0].detail}` : ''}`)
  if (!env.asrModel) warnings.push('没有找到 Qwen3-ASR 权重目录：ASR 不可用。用 asrModelPath 指定。')
  if (!env.ollama.reachable) warnings.push(`Ollama 不可达：${env.ollama.error}`)
  if (mode === 'standard' && !visionModel) warnings.push('没有可用的视觉模型：标准档会退化成"只转写 + 确定性画面信号"。')
  if (!textModel) warnings.push('没有可用的文本模型：时间线只能给原始的带时间码转写。')
  if (mode === 'standard' && frameCount > 160) {
    warnings.push(`粗筛帧 ${frameCount} 张，逐帧描述预计 ${Math.round((frameCount * 3.2) / 60)} 分钟；可调大 coarse_sec。`)
  }
  const budget = estimate(duration, frameCount, mode, settings)
  const plan = {
    version: 1,
    planId: '',
    createdAt: new Date().toISOString(),
    mode,
    input: {
      path: input.path,
      key: videoKey(input.path),
      size: input.size,
      duration: Number(duration.toFixed(2)),
      width: probeInfo.width,
      height: probeInfo.height,
      fps: probeInfo.fps,
      hasAudio: probeInfo.hasAudio,
      audioCodec: probeInfo.audioCodec,
    },
    settings: {
      coarseSec,
      coarseWidth: settings.coarseWidth,
      sceneThreshold: settings.sceneThreshold,
      maxFrames: settings.maxFrames,
      captionPerCall: settings.captionPerCall,
      keepFrames: settings.keepFrames,
      language: settings.asrLanguage,
      asrTargetSec: settings.asrTargetSec,
      keyFrameWidth: settings.keyFrameWidth,
    },
    models: {
      asr: {
        python: env.python ?? '',
        model: env.asrModel ?? '',
        language: String(args.language ?? settings.asrLanguage),
        batch: settings.asrBatch,
        threads: settings.asrThreads,
      },
      vision: { model: visionModel, available: Boolean(visionModel), enabled: mode !== 'fast' },
      text: { model: textModel, available: Boolean(textModel) },
      ffmpeg: env.ffmpeg ?? '',
      ffprobe: env.ffprobe ?? '',
      ollama: env.ollama.reachable ? settings.ollamaBaseURL : '',
    },
    budget: {
      frames: frameCount,
      ...budget,
      note: '粗略估算：按本机实测中位值（ASR 约 2.4× 实时、GPU 每帧约 4 秒、导出约 3 秒）推算，不同机器可能差数倍，仅用于判断规模。',
    },
    warnings,
  }
  const stable = JSON.stringify({
    input: plan.input,
    mode: plan.mode,
    settings: plan.settings,
    models: plan.models,
  })
  plan.planId = createHash('sha1').update(stable).digest('hex').slice(0, 16)
  const dir = stateDir(root, input.path)
  mkdirSync(dir, { recursive: true })
  writeJson(join(dir, 'plan.json'), plan)
  writeJson(join(dir, 'probe.json'), probeInfo)
  return { plan, dir, probeInfo, env }
}

export function loadPlan(root, file) {
  const dir = stateDir(root, file)
  const plan = readJson(join(dir, 'plan.json'), undefined)
  if (!plan) {
    throw new Error('还没有本次任务的计划：先调用 video_local_plan（它会报告将使用的模型与预计耗时，等你确认后再跑 scan）。')
  }
  return { plan, dir }
}

/* ------------------------------------------------------------------- scan */

export async function scan(root, inputArg, args, settings, env, options = {}) {
  const { signal, onProgress = () => {} } = options
  const input = resolveInput(root, inputArg)
  const { plan, dir } = loadPlan(root, input.path)
  if (plan.input.key !== videoKey(input.path)) {
    throw new Error('视频文件在计划之后发生了变化（大小或修改时间不同）：请重新调用 video_local_plan。')
  }
  if (args.plan_id && String(args.plan_id) !== plan.planId) {
    throw new Error(`plan_id 不匹配：当前计划是 ${plan.planId}。模型或参数已经变化，请重新 plan 并确认。`)
  }
  if (args.confirm_models !== true) {
    throw new Error(
      `按约定，第一次动用本地模型前必须确认本次选择：ASR=${plan.models.asr.model || '(不可用)'} · 视觉=${plan.models.vision.enabled ? plan.models.vision.model || '(不可用)' : '不启用（快速档）'} · 文本=${plan.models.text.model || '(不可用)'}。确认后带 confirm_models: true 再调用。`,
    )
  }
  const started = Date.now()
  const statePath = join(dir, 'state.json')
  const state = readJson(statePath, { stages: {} })
  const tmp = join(dir, 'tmp')
  mkdirSync(tmp, { recursive: true })
  mkdirSync(join(dir, 'frames'), { recursive: true })

  onProgress({ stage: 'probe' })
  const probeInfo = await probe(env.ffprobe, input.path, { cwd: root, signal })
  writeJson(join(dir, 'probe.json'), probeInfo)

  onProgress({ stage: 'scenes' })
  let scenes = []
  try {
    scenes = await detectScenes(env.ffmpeg, input.path, {
      threshold: settings.sceneThreshold,
      signal,
      cwd: root,
    })
  } catch (error) {
    onProgress({ stage: 'scenes', warning: String(error?.message ?? error) })
  }
  writeJson(join(dir, 'scenes.json'), { threshold: settings.sceneThreshold, cuts: scenes })

  const audioPath = join(dir, 'audio.wav')
  const transcriptPath = join(dir, 'transcript.json')
  let transcript = { chunks: [], audioSeconds: 0, elapsedSeconds: 0 }
  let asrCached = false
  if (probeInfo.hasAudio && env.python && plan.models.asr.model) {
    const previous = args.force === true ? undefined : readJson(transcriptPath, undefined)
    const reusable =
      previous &&
      Array.isArray(previous.chunks) &&
      previous.chunks.length > 0 &&
      Math.abs(Number(previous.audioSeconds ?? 0) - Number(probeInfo.duration)) < 1.5 &&
      // plan ids cover model + language + chunking: a changed plan must re-run.
      state.stages?.scan?.planId === plan.planId
    if (reusable) {
      transcript = previous
      asrCached = true
      onProgress({
        stage: 'asr',
        cached: true,
        detail: '复用上次转写（plan 未变化；要重新识别请带 force: true）',
      })
    } else {
      onProgress({ stage: 'audio' })
      await extractAudio(env.ffmpeg, input.path, audioPath, { cwd: root, signal })
      onProgress({ stage: 'asr', detail: '本地 ASR 进行中（CPU，约 2.4× 实时）' })
      const outPath = transcriptPath
      const bridgePath = join(dirname(fileURLToPath(import.meta.url)), '..', 'assets', 'asr_bridge.py')
      if (!existsSync(bridgePath)) throw new Error(`ASR 桥脚本缺失：${bridgePath}`)
      await run(
        env.python,
        [
          bridgePath,
          '--model', plan.models.asr.model,
          '--wav', audioPath,
          '--out', outPath,
          '--language', plan.models.asr.language,
          '--threads', String(plan.models.asr.threads || 0),
          '--target-sec', String(settings.asrTargetSec),
          '--batch', String(plan.models.asr.batch || 4),
          '--max-tokens', '448',
        ],
        {
          cwd: root,
          signal,
          timeoutMs: 4 * 60 * 60 * 1000,
          maxOutput: 4 * 1024 * 1024,
          env: {
            ...process.env,
            TEMP: tmp,
            TMP: tmp,
            NUMBA_CACHE_DIR: join(tmp, 'numba'),
            PYTHONIOENCODING: 'utf-8',
            PYTHONUNBUFFERED: '1',
          },
        },
      )
      transcript = readJson(outPath, transcript)
    }
  } else {
    onProgress({
      stage: 'asr',
      detail: probeInfo.hasAudio
        ? '没有可用的 Python/ASR 权重，跳过 ASR（计划里已有警告）'
        : '没有音轨，跳过 ASR',
    })
  }

  onProgress({ stage: 'frames' })
  const frameDir = join(dir, 'frames')
  const frames = await extractIntervalFrames(env.ffmpeg, input.path, frameDir, {
    stepSec: Number(plan.settings?.coarseSec ?? settings.coarseSec),
    width: Number(plan.settings?.coarseWidth ?? settings.coarseWidth),
    signal,
    cwd: root,
    limit: Number(plan.settings?.maxFrames ?? settings.maxFrames),
  })

  const stats = {
    duration: Number(probeInfo.duration.toFixed(2)),
    audioSeconds: transcript.audioSeconds ?? 0,
    asrSeconds: transcript.elapsedSeconds ?? 0,
    asrCached,
    chunks: transcript.chunks?.length ?? 0,
    characters: (transcript.chunks ?? []).reduce((sum, chunk) => sum + String(chunk.text ?? '').length, 0),
    scenes: scenes.length,
    frames: frames.length,
    elapsedSeconds: Math.round((Date.now() - started) / 1000),
  }
  state.stages.scan = { at: new Date().toISOString(), stats, planId: plan.planId }
  writeJson(statePath, state)
  writeJson(join(dir, 'frames.json'), { stepSec: settings.coarseSec, frames })
  return { plan, dir, probe: probeInfo, scenes, frames, transcript, stats, state }
}

/* ---------------------------------------------------------------- caption */

export async function caption(root, inputArg, args, settings, env, options = {}) {
  const { signal, onProgress = () => {} } = options
  const input = resolveInput(root, inputArg)
  const { plan, dir } = loadPlan(root, input.path)
  if (plan.mode === 'fast') {
    throw new Error('当前计划是快速档（不跑视觉模型）。需要帧描述请重新 plan，mode 用 standard。')
  }
  const visionModel = String(args.vision_model ?? plan.models.vision.model ?? '')
  if (!visionModel) throw new Error('没有可用的视觉模型：先 ollama pull 一个带 vision 能力的模型，或重新 plan。')
  if (!env.ollama.reachable) throw new Error(`Ollama 不可达：${env.ollama.error}`)

  const framesFile = join(dir, 'frames.json')
  const framesData = readJson(framesFile, undefined)
  if (!framesData) throw new Error('还没有抽好的粗筛帧：先调用 video_local_scan。')
  const frames = framesData.frames ?? []
  const captionsPath = join(dir, 'captions.json')
  const store = readJson(captionsPath, { frames: [], model: visionModel })
  const done = new Map((store.frames ?? []).map((entry) => [entry.index, entry]))
  if (args.force === true) done.clear()

  const limit = Math.max(1, Number(args.limit ?? settings.captionPerCall))
  const pending = frames.filter((frame) => !done.has(frame.index)).slice(0, limit)
  const started = Date.now()
  let processed = 0
  const failures = []
  for (const frame of pending) {
    if (signal?.aborted) throw new Error('cancelled')
    onProgress({ stage: 'caption', done: processed, total: pending.length, at: frame.t })
    try {
      const base64 = readFileSync(frame.path).toString('base64')
      const described = await describeFrame(settings.ollamaBaseURL, {
        model: visionModel,
        image: base64,
        at: frame.t,
        signal,
        prompt: framePrompt(frame.t),
      })
      done.set(frame.index, {
        index: frame.index,
        at: frame.t,
        caption: described.caption,
        importance: described.importance,
        tags: described.tags,
        ms: described.ms,
        hadThinking: described.hadThinking,
        path: frame.path,
      })
      store.frames = [...done.values()].sort((a, b) => a.index - b.index)
      store.model = visionModel
      store.updatedAt = new Date().toISOString()
      processed += 1
      if (processed % 5 === 0) writeJson(captionsPath, store)
    } catch (error) {
      if (/cancelled/i.test(String(error?.message))) throw new Error('cancelled')
      failures.push({ index: frame.index, at: frame.t, error: String(error?.message ?? error).slice(0, 200) })
      if (failures.length >= 3) break
    }
  }
  writeJson(captionsPath, store)
  const remaining = frames.length - done.size
  const top = (store.frames ?? [])
    .slice()
    .sort((a, b) => b.importance - a.importance || a.at - b.at)
    .slice(0, 8)
  return {
    dir,
    model: visionModel,
    total: frames.length,
    captioned: done.size,
    processed,
    remaining,
    failures,
    elapsedSeconds: Math.round((Date.now() - started) / 1000),
    top,
  }
}

function framePrompt(at) {
  return [
    `这是视频第 ${formatClock(at)} 秒的画面。`,
    '用一句中文描述画面（不超过40字）：什么软件/界面、关键文字、人物在做什么。',
    '再给这一帧对理解视频的重要程度：1=几乎无变化的过渡，3=一般，5=关键信息（新界面、重要文字、演示步骤、结果）。',
    '只输出 JSON：{"caption":"","importance":3,"tags":[""]}，不要解释。',
  ].join('\n')
}

/* --------------------------------------------------------------- timeline */

export async function buildTimeline(root, inputArg, args, settings, env, options = {}) {
  const { signal, onProgress = () => {} } = options
  const input = resolveInput(root, inputArg)
  const { plan, dir } = loadPlan(root, input.path)
  const transcript = readJson(join(dir, 'transcript.json'), undefined)
  if (!transcript) throw new Error('还没有转写：先调用 video_local_scan。')
  const chunks = (transcript.chunks ?? []).filter((chunk) => String(chunk.text ?? '').trim())
  const captions = readJson(join(dir, 'captions.json'), { frames: [] }).frames ?? []
  if (chunks.length === 0 && captions.length === 0) {
    throw new Error('转写为空（视频可能没有音轨或没有可识别语音），也没有帧描述：先跑 video_local_caption，或换一支有语音的视频。')
  }
  const scenes = readJson(join(dir, 'scenes.json'), { cuts: [] }).cuts ?? []
  const textModel = String(args.text_model ?? plan.models.text.model ?? '')
  const started = Date.now()

  // A silent video still gets a timeline: the captions become the narrative.
  // A transcript that is technically non-empty but contentless ("嗯。") must not
  // outrank real frame descriptions, so judge it by speech density too.
  const speechChars = chunks.reduce((sum, chunk) => sum + String(chunk.text ?? '').trim().length, 0)
  const duration = Number(plan.input.duration ?? 0)
  const uniqueChars = new Set(chunks.map((chunk) => String(chunk.text ?? '')).join('')).size
  const speechUsable = chunks.length > 0 && (speechChars >= Math.max(24, duration * 0.8) || uniqueChars >= 30)
  const narrative = speechUsable ? chunks : captionNarrative(captions, settings)
  const narrativeSource = speechUsable ? 'transcript' : 'captions'
  if (narrative.length === 0) {
    throw new Error('台词太少、也没有帧描述可用：至少先跑 video_local_caption。')
  }
  const windows = chunkWindow(narrative, settings.textWindowChars)
  const notes = []
  if (textModel && env.ollama.reachable) {
    for (let index = 0; index < windows.length; index++) {
      if (signal?.aborted) throw new Error('cancelled')
      onProgress({ stage: 'timeline', done: index, total: windows.length })
      const window = windows[index]
      const prompt = [
        '下面是一个视频里连续几段带时间码的台词或画面描述。请归纳这一段讲了什么。',
        '只输出 JSON：{"title":"本段小标题(不超过12字)","summary":"一句话概括(不超过60字)","headings":[{"start":"mm:ss","title":"小标题"}],"keywords":[""]}',
        'headings 的时间必须是下面这些时间码范围之内的时刻，最多 3 条；keywords 最多 6 个；全部用中文，不要解释。',
        '',
        window.map((chunk) => `[${formatClock(chunk.start)}-${formatClock(chunk.end)}] ${chunk.text}`).join('\n'),
      ].join('\n')
      try {
        const answer = await chat(settings.ollamaBaseURL, {
          model: textModel,
          prompt,
          format: 'json',
          think: false,
          signal,
          timeoutMs: 240000,
          options: { num_predict: 600, num_ctx: 8192, temperature: 0.2 },
        })
        const parsed = parseJsonLoose(answer.content) ?? {}
        const from = window[0].start
        const to = window[window.length - 1].end
        const headings = Array.isArray(parsed.headings)
          ? parsed.headings
              .map((heading) => ({ start: parseClock(heading?.start), title: String(heading?.title ?? '').trim() }))
              .filter((heading) => heading.title && Number.isFinite(heading.start))
              .filter((heading) => heading.start >= from - 2 && heading.start <= to + 2)
              .sort((a, b) => a.start - b.start)
              .slice(0, 3)
          : []
        notes.push({
          from,
          to,
          title: String(parsed.title ?? '').trim(),
          summary: String(parsed.summary ?? '').trim() || window.map((c) => c.text).join(' ').slice(0, 160),
          headings,
          keywords: Array.isArray(parsed.keywords) ? parsed.keywords.map(String).filter(Boolean).slice(0, 8) : [],
        })
      } catch (error) {
        if (/cancelled/i.test(String(error?.message))) throw new Error('cancelled')
        notes.push({
          from: window[0].start,
          to: window[window.length - 1].end,
          title: '',
          summary: window.map((c) => c.text).join(' ').slice(0, 220),
          headings: [],
          keywords: [],
          error: String(error?.message ?? error).slice(0, 200),
        })
      }
    }
  } else {
    for (const window of windows) {
      notes.push({
        from: window[0].start,
        to: window[window.length - 1].end,
        summary: window.map((chunk) => chunk.text).join(' ').slice(0, 220),
        headings: [],
        keywords: [],
      })
    }
  }

  const outline = notes.map((note, index) => ({
    index,
    start: Number(note.from.toFixed(1)),
    end: Number(note.to.toFixed(1)),
    title: String(note.title || '').trim() || note.headings?.[0]?.title || `第 ${index + 1} 段`,
    summary: note.summary,
    headings: note.headings ?? [],
    keywords: note.keywords ?? [],
  }))

  const keywords = [...new Set(outline.flatMap((segment) => segment.keywords))].slice(0, 24)
  const frameCount = readJson(join(dir, 'frames.json'), { frames: [] }).frames?.length ?? 0
  const coverage = {
    captioned: captions.length,
    frames: frameCount,
    complete: frameCount > 0 && captions.length >= frameCount,
  }
  const keyFrames = pickKeyFrames({
    captions,
    scenes,
    outline,
    transcript: chunks,
    limit: Number(args.frames ?? settings.keepFrames),
    perSegment: Number(settings.framesPerSegment ?? 2),
  })
  const result = {
    version: 1,
    generatedAt: new Date().toISOString(),
    input: { path: input.path, duration: plan.input.duration },
    mode: plan.mode,
    models: {
      asr: plan.models.asr.model,
      vision: captions.length > 0 ? plan.models.vision.model : '',
      text: textModel,
    },
    title: outline[0]?.title && outline.length === 1 ? outline[0].title : basename(input.path).replace(/\.[^.]+$/, ''),
    keywords,
    outline,
    keyFrames,
    coverage,
    narrativeSource,
    speechChars,
    captionsCount: captions.length,
    scenesCount: scenes.length,
    transcript: chunks,
    stats: {
      elapsedSeconds: Math.round((Date.now() - started) / 1000),
      windows: windows.length,
      textModel: textModel || '(none)',
    },
    provenance:
      '本地模型初筛结果：ASR 与帧描述可能出错，时间线和纲要只用于缩小范围，不是最终结论。最终判断请直接看导出的原始关键帧。',
  }
  writeJson(join(dir, 'timeline.json'), result)
  writeFileSync(join(dir, 'timeline.md'), renderMarkdown(result, plan, dir), 'utf8')
  return { dir, timeline: result }
}

function captionNarrative(captions, settings) {
  const step = Number(settings.coarseSec ?? 5)
  return (captions ?? [])
    .filter((entry) => String(entry.caption ?? '').trim())
    .map((entry) => ({ start: entry.at, end: entry.at + step, text: `[画面] ${entry.caption}` }))
}

function chunkWindow(chunks, limit) {
  const windows = []
  let current = []
  let size = 0
  for (const chunk of chunks) {
    const text = String(chunk.text ?? '')
    if (current.length > 0 && size + text.length > limit) {
      windows.push(current)
      current = []
      size = 0
    }
    current.push(chunk)
    size += text.length
  }
  if (current.length > 0) windows.push(current)
  return windows
}

export function parseClock(value) {
  const match = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(String(value ?? '').trim())
  if (!match) return undefined
  const first = Number(match[1])
  const second = Number(match[2])
  const third = match[3] === undefined ? undefined : Number(match[3])
  return third === undefined ? first * 60 + second : first * 3600 + second * 60 + third
}

function pickKeyFrames({ captions, scenes, outline, transcript, limit, perSegment }) {
  const cutNear = (at) => scenes.some((cut) => Math.abs(cut - at) <= 2)
  const keywordNear = (at) => {
    const nearby = transcript.filter((chunk) => at >= chunk.start - 12 && at <= chunk.end + 12)
    const text = nearby.map((chunk) => chunk.text).join(' ')
    return outline.flatMap((segment) => segment.keywords).some((keyword) => keyword && text.includes(keyword))
  }
  const scored = []
  for (const entry of captions ?? []) {
    if (!Number.isFinite(entry.at)) continue
    if (!String(entry.caption ?? '').trim()) continue
    let score = Number(entry.importance ?? 0) * 2
    if (cutNear(entry.at)) score += 0.8
    if (keywordNear(entry.at)) score += 0.5
    scored.push({ ...entry, score })
  }
  scored.sort((a, b) => b.score - a.score || a.at - b.at)

  const segments = Math.max(1, outline.length)
  // Spread over chapters, but never let the per-chapter cap alone decide the
  // final count: a one-chapter video must still be able to use the whole budget.
  const cap = Math.max(2, Math.ceil(limit / segments))
  const segmentOf = (at) => {
    const index = outline.findIndex((segment) => at >= segment.start - 1 && at <= segment.end + 1)
    return index < 0 ? -1 : index
  }
  const picked = []
  const perSegmentCount = new Map()
  const fingerprint = (entry) => String(entry.caption ?? '').replace(/[\s，。,.、：:；;]/g, '').slice(0, 14)
  const push = (entry, enforceCap) => {
    if (picked.length >= limit) return false
    if (picked.some((existing) => Math.abs(existing.at - entry.at) < 8)) return false
    if (fingerprint(entry) && picked.some((existing) => fingerprint(existing) === fingerprint(entry) && Math.abs(existing.at - entry.at) < 30)) return false
    if (enforceCap) {
      const index = segmentOf(entry.at)
      if (index >= 0 && (perSegmentCount.get(index) ?? 0) >= cap) return false
      perSegmentCount.set(index, (perSegmentCount.get(index) ?? 0) + 1)
    }
    picked.push(entry)
    return true
  }
  // First make sure every chapter is represented, then fill by global score.
  for (const segment of outline) {
    if (picked.length >= limit) break
    const inside = scored.filter((entry) => entry.at >= segment.start - 1 && entry.at <= segment.end + 1)
    let taken = 0
    for (const candidate of inside) {
      if (taken >= Math.max(2, Math.floor(limit / segments)) || picked.length >= limit) break
      if (push(candidate, true)) taken += 1
    }
  }
  for (const candidate of scored) {
    if (picked.length >= limit) break
    push(candidate, true)
  }
  picked.sort((a, b) => a.at - b.at)
  return picked.map((entry) => ({
    at: entry.at,
    caption: entry.caption,
    importance: entry.importance,
    score: Number(entry.score.toFixed(2)),
    tags: entry.tags ?? [],
    frame: entry.path ?? '',
    reason:
      [cutNear(entry.at) ? '场景切点附近' : '', keywordNear(entry.at) ? '台词关键词命中' : '']
        .filter(Boolean)
        .join(' · ') || '画面重要度最高',
  }))
}

function renderMarkdown(result, plan, dir) {
  const lines = []
  lines.push(`# ${result.title}`)
  lines.push('')
  lines.push('> ⚠️ **未核验的导航文本**：以下时间线、动作描述由本地小模型生成，可能与画面不符。**结论请以 `frames/` 里的原帧为准**；只读这一页容易把描述当成事实。')
  lines.push('')
  lines.push(`- 来源：\`${result.input.path}\``)
  lines.push(`- 时长：${formatTime(result.input.duration)}`)
  lines.push(`- 本次模型：ASR \`${result.models.asr || '(未用)'}\` · 帧描述 \`${result.models.vision || '(未用)'}\` · 归纳 \`${result.models.text || '(未用)'}\``)
  if (result.narrativeSource === 'captions') {
    lines.push(`- 台词过少（${result.speechChars ?? 0} 字），下面的时间线由**画面描述**归纳，不是台词。`)
  }
  if (result.coverage && !result.coverage.complete) {
    lines.push(
      `- 关键帧覆盖：只描述了 ${result.coverage.captioned}/${result.coverage.frames} 帧，下面的关键帧候选只来自已经描述过的帧（继续调用 video_local_caption 可以补全）。`,
    )
  }
  lines.push(`- 生成时间：${result.generatedAt}`)
  if (result.keywords.length > 0) lines.push(`- 关键词：${result.keywords.join('、')}`)
  lines.push('')
  if (result.keyFrames.length > 0) {
    lines.push('## 关键帧候选')
    lines.push('')
    for (const frame of result.keyFrames) {
      lines.push(`- **${formatClock(frame.at)}**（重要度 ${frame.importance}/5，${frame.reason}）${frame.caption}`)
    }
    lines.push('')
  }
  lines.push('## 时间线')
  lines.push('')
  for (const segment of result.outline) {
    lines.push(`### ${formatClock(segment.start)} – ${formatClock(segment.end)}　${segment.title}`)
    lines.push('')
    lines.push(segment.summary)
    if (Array.isArray(segment.headings) && segment.headings.length > 1) {
      const extra = segment.headings.filter(
        (heading) => heading.title && heading.title !== segment.title && heading.start > segment.start + 2 && heading.start <= segment.end,
      )
      if (extra.length > 0) {
        lines.push('')
        for (const heading of extra) lines.push(`- ${formatClock(heading.start)}　${heading.title}`)
      }
    }
    const inSegment = result.keyFrames.filter((frame) => frame.at >= segment.start - 2 && frame.at <= segment.end + 2)
    if (inSegment.length > 0) {
      lines.push('')
      for (const frame of inSegment) {
        lines.push(`- ${formatClock(frame.at)}　${frame.caption}`)
      }
    }
    lines.push('')
  }
  lines.push('## 带时间码转写（ASR）')
  lines.push('')
  for (const chunk of result.transcript) {
    lines.push(`- \`${formatClock(chunk.start)}–${formatClock(chunk.end)}\` ${chunk.text}`)
  }
  lines.push('')
  return `${lines.join('\n')}\n`
}

/* ----------------------------------------------------------------- export */

export async function exportAll(root, inputArg, args, settings, env, options = {}) {
  const { signal, onProgress = () => {} } = options
  const input = resolveInput(root, inputArg)
  const { plan, dir } = loadPlan(root, input.path)
  const timeline = readJson(join(dir, 'timeline.json'), undefined)
  if (!timeline) throw new Error('还没有时间线：先调用 video_local_timeline。')
  const started = Date.now()
  const outDir = join(dir, 'export')
  const framesDir = join(outDir, 'frames')
  mkdirSync(framesDir, { recursive: true })

  const count = Math.max(1, Number(args.frames ?? timeline.keyFrames.length ?? 8))
  const selected = (timeline.keyFrames ?? []).slice(0, count)
  const written = []
  for (const [index, frame] of selected.entries()) {
    if (signal?.aborted) throw new Error('cancelled')
    onProgress({ stage: 'export', done: index, total: selected.length })
    const name = `key-${String(index + 1).padStart(2, '0')}-${Math.round(frame.at)}s.jpg`
    const target = join(framesDir, name)
    try {
      await extractFrameAt(env.ffmpeg, input.path, frame.at, target, {
        width: settings.keyFrameWidth,
        cwd: root,
        signal,
      })
      written.push({ ...frame, path: target })
    } catch (error) {
      onProgress({ stage: 'export', warning: String(error?.message ?? error) })
    }
  }

  // Filler-only audio ("嗯。") makes a subtitle file that is worse than none.
  const speechUsableForSubtitles = timeline.narrativeSource !== 'captions'
  const srtPath = join(outDir, 'subtitles.srt')
  writeFileSync(srtPath, speechUsableForSubtitles ? buildSrt(timeline.transcript ?? []) : '', 'utf8')
  const mdPath = join(outDir, 'timeline.md')
  writeFileSync(mdPath, renderMarkdown(timeline, plan, dir), 'utf8')
  writeJson(join(outDir, 'timeline.json'), { ...timeline, exportedFrames: written })
  const reportPath = join(outDir, 'report.md')
  writeFileSync(
    reportPath,
    [
      `# ${timeline.title}`,
      '',
      '> ⚠️ **未核验**：以下描述由本地小模型生成，可能与画面不符。结论请直接看 `frames/` 里的原帧。',
      '',
      `- 视频：\`${input.path}\``,
      `- 时长：${formatTime(timeline.input.duration)}`,
      `- 模型：ASR \`${timeline.models.asr || '(未用)'}\` · 帧描述 \`${timeline.models.vision || '(未用)'}\` · 归纳 \`${timeline.models.text || '(未用)'}\``,
      `- 关键帧覆盖：${timeline.coverage ? `${timeline.coverage.captioned}/${timeline.coverage.frames} 帧已描述${timeline.coverage.complete ? '' : '（不完整，候选只来自已描述帧）'}` : '未知'}`,
      `- 时间线来源：${timeline.narrativeSource === 'captions' ? `画面描述（台词只有 ${timeline.speechChars ?? 0} 字，不可用）` : 'ASR 台词'}`,
      `- 字幕：${speechUsableForSubtitles ? `\`${basename(srtPath)}\`（${timeline.transcript.length} 条）` : '跳过（台词不可用，不生成空字幕）'}`,
      `- 关键帧：${written.length} 张，见 \`frames/\``,
      `- 导出耗时：${Math.round((Date.now() - started) / 1000)} 秒`,
      '',
      '## 关键帧清单',
      '',
      ...written.map((frame) => `- ${formatClock(frame.at)}（重要度 ${frame.importance}/5）${frame.caption} → \`frames/${basename(frame.path)}\``),
      '',
    ].join('\n'),
    'utf8',
  )
  const statePath = join(dir, 'state.json')
  const state = readJson(statePath, { stages: {} })
  state.stages.export = { at: new Date().toISOString(), frames: written.length }
  writeJson(statePath, state)
  return { dir, outDir, srtPath, mdPath, reportPath, frames: written, timeline }
}

/** Sentence-level SRT from ASR chunks; time is shared by character count. */
export function buildSrt(chunks) {
  const cues = []
  let index = 1
  for (const chunk of chunks ?? []) {
    const text = String(chunk.text ?? '').trim()
    if (!text) continue
    const sentences = text.split(/(?<=[。！？!?；;])\s*/).map((part) => part.trim()).filter(Boolean)
    const parts = sentences.length > 0 ? sentences : [text]
    const total = parts.reduce((sum, part) => sum + part.length, 0) || parts.length
    const duration = Math.max(0.5, Number(chunk.end) - Number(chunk.start))
    let cursor = Number(chunk.start)
    for (const part of parts) {
      const share = Math.max(0.5, (part.length / total) * duration)
      const start = cursor
      const end = Math.min(Number(chunk.end) + 0.4, cursor + share)
      cues.push({ index: index++, start, end: Math.max(end, start + 0.4), text: part })
      cursor = end
    }
  }
  return `${cues.map((cue) => `${cue.index}\n${srtTime(cue.start)} --> ${srtTime(cue.end)}\n${cue.text}\n`).join('\n')}`
}

export function srtTime(seconds) {
  const total = Math.max(0, Number(seconds) || 0)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = Math.floor(total % 60)
  const ms = Math.round((total - Math.floor(total)) * 1000)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`
}

/* ----------------------------------------------------------------- frames */

export async function grabFrames(root, inputArg, args, settings, env, options = {}) {
  const { signal, onProgress = () => {} } = options
  const input = resolveInput(root, inputArg)
  const dir = stateDir(root, input.path)
  const probeInfo = readJson(join(dir, 'probe.json'), undefined) ?? (await probe(env.ffprobe, input.path, { cwd: root }))
  const duration = Number(probeInfo.duration ?? 0)
  const start = Math.max(0, Number(args.start ?? 0))
  const end = Math.min(duration > 0 ? duration : start + 10, Number(args.end ?? start + 10))
  if (!(end > start)) throw new Error(`时间段不合法：start=${start} end=${end}`)
  const count = Math.max(1, Math.min(12, Number(args.count ?? 4)))
  const width = Math.max(320, Number(args.width ?? settings.keyFrameWidth))
  const outDir = join(dir, 'grab')
  mkdirSync(outDir, { recursive: true })
  const step = (end - start) / count
  const frames = []
  for (let index = 0; index < count; index++) {
    if (signal?.aborted) throw new Error('cancelled')
    const at = Number((start + step * index + Math.min(step / 2, 0.5)).toFixed(2))
    onProgress({ stage: 'frames', done: index, total: count, at })
    const target = join(outDir, `grab-${Date.now()}-${index}.jpg`)
    await extractFrameAt(env.ffmpeg, input.path, at, target, { width, cwd: root, signal })
    frames.push({ at, path: target })
  }
  return { dir, frames, start, end, width }
}

/* ------------------------------------------------------------- utilities */

export function copyFrame(source, target) {
  mkdirSync(dirname(target), { recursive: true })
  copyFileSync(source, target)
  return target
}

export function listFrameFiles(dir) {
  try {
    return readdirSync(dir).filter((name) => /\.jpe?g$/i.test(name))
  } catch {
    return []
  }
}
