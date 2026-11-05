/**
 * Deterministic media layer: probe one video, pull its audio, find scene cuts
 * and extract JPEG frames. Everything here is ffmpeg/ffprobe only, so the
 * expensive local models never run before the cheap signals exist.
 */
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'

export const FFMPEG_CANDIDATES = [
  'D:\\Program files\\FFmpeg\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe',
  'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
  '/usr/bin/ffmpeg',
  '/usr/local/bin/ffmpeg',
  '/opt/homebrew/bin/ffmpeg',
]

export const FFPROBE_CANDIDATES = FFMPEG_CANDIDATES.map((path) => path.replace(/ffmpeg(\.exe)?$/i, (m) => (m.toLowerCase().endsWith('.exe') ? 'ffprobe.exe' : 'ffprobe')))

/** Run one child process, capture bounded output, honour the abort signal. */
export function run(command, args, options = {}) {
  const {
    cwd,
    env,
    signal,
    timeoutMs = 60 * 60 * 1000,
    maxOutput = 8 * 1024 * 1024,
    stdin,
  } = options
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    let settled = false
    const finish = (fn) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      signal?.removeEventListener?.('abort', onAbort)
      fn()
    }
    const timer = setTimeout(() => {
      child.kill()
      finish(() => reject(new Error(`timeout after ${Math.round(timeoutMs / 1000)}s: ${command}`)))
    }, timeoutMs)
    const onAbort = () => {
      child.kill()
      finish(() => reject(new Error('cancelled')))
    }
    if (signal?.aborted) return onAbort()
    signal?.addEventListener?.('abort', onAbort, { once: true })
    child.stdout?.on('data', (chunk) => {
      if (stdout.length < maxOutput) stdout += chunk.toString('utf8')
    })
    child.stderr?.on('data', (chunk) => {
      if (stderr.length < maxOutput) stderr += chunk.toString('utf8')
    })
    child.on('error', (error) => {
      const hint =
        error?.code === 'EPERM'
          ? '（spawn EPERM：当前进程无法用管道启动子进程。插件需要运行在不受限的 Host 进程里；受限沙箱只允许 stdio inherit/ignore 的 spawn。）'
          : ''
      finish(() => reject(new Error(`${String(error?.message ?? error)}${hint}`)))
    })
    child.on('close', (code) => {
      finish(() => {
        if (code === 0) resolve({ code, stdout, stderr })
        else reject(new Error(fmtFailure(command, args, code, stderr, stdout)))
      })
    })
    if (stdin !== undefined) child.stdin?.end(stdin)
    else child.stdin?.end()
  })
}

function fmtFailure(command, args, code, stderr, stdout) {
  const tail = String(stderr || stdout || '').trim().split(/\r?\n/).slice(-6).join('\n')
  const name = String(command).split(/[\\/]/).pop()
  return `${name} exited ${code}: ${tail || '(no output)'}`
}

export function which(binary, candidates = []) {
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) return candidate
  }
  return findOnPath(binary)
}

function findOnPath(binary) {
  const path = process.env.PATH ?? ''
  const exts = process.platform === 'win32' ? (process.env.PATHEXT ?? '.EXE;.CMD;.BAT').split(';') : ['']
  for (const dir of path.split(process.platform === 'win32' ? ';' : ':')) {
    if (!dir) continue
    for (const ext of exts) {
      const file = join(dir, binary + ext)
      try {
        if (statSync(file).isFile()) return file
      } catch {
        /* keep looking */
      }
    }
  }
  return undefined
}

/** ffprobe one file into the facts the rest of the pipeline needs. */
export async function probe(ffprobe, file, options = {}) {
  const { signal, cwd } = options
  const { stdout } = await run(
    ffprobe,
    [
      '-v', 'error',
      '-print_format', 'json',
      '-show_format',
      '-show_streams',
      file,
    ],
    { signal, cwd, timeoutMs: 120000 },
  )
  const payload = JSON.parse(stdout)
  const streams = Array.isArray(payload.streams) ? payload.streams : []
  const video = streams.find((s) => s.codec_type === 'video')
  const audio = streams.find((s) => s.codec_type === 'audio')
  const duration = Number(payload.format?.duration ?? video?.duration ?? 0)
  const [num, den] = String(video?.r_frame_rate ?? '0/1').split('/').map(Number)
  return {
    path: file,
    duration,
    size: Number(payload.format?.size ?? 0),
    bitRate: Number(payload.format?.bit_rate ?? 0),
    width: Number(video?.width ?? 0),
    height: Number(video?.height ?? 0),
    fps: den ? Number((num / den).toFixed(3)) : 0,
    videoCodec: String(video?.codec_name ?? ''),
    hasAudio: Boolean(audio),
    audioCodec: String(audio?.codec_name ?? ''),
    audioSampleRate: Number(audio?.sample_rate ?? 0),
  }
}

/** Extract the whole soundtrack as 16 kHz mono PCM16, the ASR bridge's input. */
export async function extractAudio(ffmpeg, file, outWav, options = {}) {
  mkdirSync(dirname(outWav), { recursive: true })
  await run(
    ffmpeg,
    [
      '-hide_banner', '-nostdin', '-v', 'error', '-y',
      '-i', file,
      '-vn',
      '-ac', '1',
      '-ar', '16000',
      '-c:a', 'pcm_s16le',
      outWav,
    ],
    { signal: options.signal, cwd: options.cwd, timeoutMs: 30 * 60 * 1000 },
  )
  return outWav
}

/** Scene-cut timestamps in seconds, from ffmpeg's scene score. */
export async function detectScenes(ffmpeg, file, options = {}) {
  const { threshold = 0.28, signal, cwd, limit = 600 } = options
  const { stderr } = await run(
    ffmpeg,
    [
      '-hide_banner', '-nostdin', '-v', 'info',
      '-i', file,
      '-an',
      '-filter:v', `select='gt(scene,${threshold})',showinfo`,
      '-f', 'null',
      process.platform === 'win32' ? 'NUL' : '/dev/null',
    ],
    { signal, cwd, timeoutMs: 60 * 60 * 1000, maxOutput: 64 * 1024 * 1024 },
  )
  const cuts = []
  const pattern = /pts_time:([0-9]+(?:\.[0-9]+)?)/g
  let match
  while ((match = pattern.exec(stderr)) !== null) {
    const at = Number(match[1])
    if (Number.isFinite(at)) cuts.push(Number(at.toFixed(2)))
  }
  const unique = [...new Set(cuts)].sort((a, b) => a - b)
  return unique.slice(0, limit)
}

/** Sample one frame every `stepSec` seconds at a fixed width. */
export async function extractIntervalFrames(ffmpeg, file, outDir, options = {}) {
  const { stepSec = 5, width = 512, signal, cwd, limit = 4000 } = options
  mkdirSync(outDir, { recursive: true })
  const pattern = join(outDir, 'c%05d.jpg')
  await run(
    ffmpeg,
    [
      '-hide_banner', '-nostdin', '-v', 'error', '-y',
      '-i', file,
      '-an',
      '-vf', `fps=1/${stepSec},scale=${width}:-2`,
      '-frames:v', String(limit),
      '-q:v', '3',
      pattern,
    ],
    { signal, cwd, timeoutMs: 60 * 60 * 1000 },
  )
  const files = readdirSync(outDir)
    .filter((name) => /^c\d+\.jpg$/.test(name))
    .sort()
  return files.map((name, index) => ({
    index,
    path: join(outDir, name),
    t: Number((index * stepSec).toFixed(2)),
  }))
}

/** One frame at an exact timestamp (fast input seek), optionally letterbox-free. */
export async function extractFrameAt(ffmpeg, file, at, outPath, options = {}) {
  const { width = 1280, signal, cwd } = options
  mkdirSync(dirname(outPath), { recursive: true })
  await run(
    ffmpeg,
    [
      '-hide_banner', '-nostdin', '-v', 'error', '-y',
      '-ss', String(Math.max(0, at)),
      '-i', file,
      '-frames:v', '1',
      '-vf', `scale='min(${width},iw)':-2`,
      '-q:v', '2',
      outPath,
    ],
    { signal, cwd, timeoutMs: 120000 },
  )
  return outPath
}

export function formatTime(seconds) {
  const total = Math.max(0, Number(seconds) || 0)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = Math.floor(total % 60)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function formatClock(seconds) {
  const total = Math.max(0, Number(seconds) || 0)
  const m = Math.floor(total / 60)
  const s = Math.floor(total % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}
