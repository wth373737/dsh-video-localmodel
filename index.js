/**
 * dsh-video-localmodel — local-first video screening for DeepSeek Harness.
 *
 * The plugin registers plain tool objects through `ctx.tools.register` and
 * imports nothing from DSH, so a harness upgrade or a pnpm relink cannot break
 * module resolution. Everything expensive happens locally and is staged:
 *
 *   video_local_plan      → probe the machine, report the exact models, wait
 *   video_local_scan      → deterministic signals + local ASR (no cloud tokens)
 *   video_local_caption   → local VLM captions with an importance score
 *   video_local_timeline  → local text model folds the transcript into chapters
 *   video_local_export    → SRT + timeline Markdown + original-resolution frames
 *   video_local_frames    → original frames for one range, for the final look
 *
 * The leading model reads the ORIGINAL candidate frames, never a local model's
 * paraphrase, and every artifact carries a provenance line saying the local
 * output is screening material rather than a conclusion.
 */
import { readFileSync } from 'node:fs'
import { basename } from 'node:path'

import {
  buildTimeline,
  caption,
  detectEnvironment,
  exportAll,
  grabFrames,
  planFor,
  resolveInput,
  resolveSettings,
  scan,
} from './lib/pipeline.js'

export const name = 'video-localmodel'
export const inject = ['tools']

const PARAM_INPUT = {
  type: 'string',
  description: 'Absolute path to a local video file (a path relative to the session workspace also works). Remote URLs are not supported in this version.',
}

const PARAM_CONFIRM = {
  type: 'boolean',
  description:
    'Must be exactly true. It is the explicit "the user saw the reported models and agreed" signal; without it the stage refuses to spend local model time.',
}

/* --------------------------------------------------------------- helpers */

async function workspaceRoot(ctx, exec) {
  const service = ctx.get?.('workingDirectory') ?? ctx.workingDirectory
  try {
    const cwd = await service?.ensure?.(exec.agent, exec.signal)
    if (typeof cwd === 'string' && cwd.length > 0) return cwd
  } catch {
    /* treat as unavailable */
  }
  return undefined
}

async function prepare(ctx, exec, inputArg, config) {
  const root = await workspaceRoot(ctx, exec)
  if (!root) throw new Error('解析不到会话工作区：无法确定产物目录（会拒绝在没有工作区时运行）。')
  const { settings, source } = resolveSettings(root, config)
  const env = await detectEnvironment(root, settings, { signal: exec.signal })
  const input = resolveInput(root, inputArg)
  return { root, settings, env, input, settingsSource: source }
}

function envSummary(env) {
  return {
    ffmpeg: env.ffmpeg ?? '',
    ffprobe: env.ffprobe ?? '',
    python: env.python ?? '',
    asr_model: env.asrModel ?? '',
    ollama: env.ollama.reachable ? env.ollama.models.length : 0,
    vision_models: env.visionCapabilities,
    text_models: env.textCapabilities,
  }
}

/** Save up to `limit` frames as durable attachments for the model to look at. */
async function attachFrames(ctx, frames, limit) {
  const attachments = ctx.get?.('attachments')
  const refs = []
  if (!attachments || typeof attachments.saveImage !== 'function') return refs
  for (const frame of frames.slice(0, limit)) {
    try {
      const ref = await attachments.saveImage({
        data: readFileSync(frame.path),
        mediaType: 'image/jpeg',
        name: basename(frame.path),
      })
      refs.push({ ref, at: frame.at ?? null, caption: frame.caption ?? '' })
    } catch {
      /* an over-limit frame must not fail the whole tool call */
    }
  }
  return refs
}

function imageBlocks(refs) {
  return refs.map((entry) => ({ type: 'image', attachment: entry.ref }))
}

function humanDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0))
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  return minutes > 0 ? `${minutes} 分 ${rest} 秒` : `${rest} 秒`
}

/* ----------------------------------------------------------------- tools */

function planTool(ctx, config) {
  return {
    name: 'video_local_plan',
    description:
      'First step for any video: probe the file, detect ffmpeg/Python/Qwen3-ASR and the Ollama models that actually exist on this machine, then return the exact model choice, warnings and a time estimate. It costs no model time and changes nothing except a plan file, so call it before video_local_scan and show the result to the user. The returned planId is what video_local_scan expects once the user agrees. 中文触发词：视频初筛计划、本地模型、看看用什么模型、视频理解。',
    parameters: {
      type: 'object',
      properties: {
        input: PARAM_INPUT,
        mode: {
          type: 'string',
          enum: ['standard', 'fast'],
          description:
            'standard (default) also captions frames with the local VLM; fast skips the vision model entirely and only does ASR + deterministic frame sampling + text outline (much cheaper).',
        },
        coarse_sec: { type: 'integer', description: 'Sample one frame every N seconds for the coarse pass (default 5).' },
        max_minutes: { type: 'integer', description: 'Refuse videos longer than this many minutes (default 30).' },
        vision_model: { type: 'string', description: 'Ollama vision model to use; defaults to the best detected one.' },
        text_model: { type: 'string', description: 'Ollama text model to use; defaults to qwen2.5:7b when installed.' },
        language: { type: 'string', description: 'ASR language, e.g. Chinese or English (default Chinese).' },
        refresh_env: { type: 'boolean', description: 'Re-detect ffmpeg/Python/models instead of using the 10-minute cache.' },
      },
      required: ['input'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          plan_id: { type: 'string' },
          mode: { type: 'string' },
          video: { type: 'object' },
          models: { type: 'object' },
          budget: { type: 'object' },
          warnings: { type: 'array', items: { type: 'string' } },
          environment: { type: 'object' },
          state_dir: { type: 'string' },
          next: { type: 'string' },
        },
        required: ['plan_id', 'mode', 'models', 'budget', 'warnings', 'next'],
      },
      render: (_args, value) => {
        const lines = []
        lines.push(`计划 ${value.plan_id}（${value.mode === 'fast' ? '快速档' : '标准档'}）`)
        lines.push(
          `视频：${value.video.path}　${value.video.duration_hms}　${value.video.width}x${value.video.height}${value.video.has_audio ? '' : '　(无音轨)'}`,
        )
        lines.push('')
        lines.push('本次将使用：')
        lines.push(`- ASR：${value.models.asr.model || '不可用'}　（Python：${value.models.asr.python || '未找到'}）`)
        lines.push(`- 视觉：${value.models.vision.available ? value.models.vision.model : '不可用/不启用'}`)
        lines.push(`- 文本归纳：${value.models.text.model || '不可用'}`)
        lines.push('')
        lines.push(
          `预计：扫描 ${humanDuration(value.budget.scanSeconds)} · 帧描述 ${humanDuration(value.budget.captionSeconds)} · 时间线 ${humanDuration(value.budget.timelineSeconds)} · 合计约 ${humanDuration(value.budget.totalSeconds)}（${value.budget.frames} 帧）`,
        )
        if (value.warnings.length > 0) {
          lines.push('')
          lines.push('需要注意：')
          for (const warning of value.warnings) lines.push(`- ${warning}`)
        }
        lines.push('')
        lines.push(value.next)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    async execute(args, exec) {
      const { root, settings, env, input } = await prepare(ctx, exec, args.input, config)
      const { plan, dir } = await planFor(root, input.path, args, settings, env)
      return {
        plan_id: plan.planId,
        mode: plan.mode,
        video: {
          path: plan.input.path,
          duration_hms: formatHms(plan.input.duration),
          duration: plan.input.duration,
          width: plan.input.width,
          height: plan.input.height,
          fps: plan.input.fps,
          has_audio: plan.input.hasAudio,
        },
        models: {
          asr: { model: plan.models.asr.model, python: plan.models.asr.python, language: plan.models.asr.language },
          vision: plan.models.vision,
          text: plan.models.text,
        },
        budget: plan.budget,
        warnings: plan.warnings,
        environment: envSummary(env),
        state_dir: dir,
        next:
          '把上面的模型选择和预计耗时告诉用户；用户同意后调用 video_local_scan，并带 plan_id 与 confirm_models: true。',
      }
    },
  }
}

function scanTool(ctx, config) {
  return {
    name: 'video_local_scan',
    description:
      'Run the cheap half of the pipeline: ffprobe, scene-cut detection, 16 kHz audio extraction and local Qwen3-ASR with per-chunk timecodes, then sample coarse frames. No cloud tokens are spent. Requires a plan made by video_local_plan and confirm_models: true, because it starts the local model work. Returns duration, chunk count, scene count, frame count, a transcript preview and the artifact directory. 中文触发词：扫描视频、抽音频、本地转写、抽帧、场景切点。',
    parameters: {
      type: 'object',
      properties: {
        input: PARAM_INPUT,
        plan_id: { type: 'string', description: 'The planId returned by video_local_plan. A mismatch means models or parameters changed.' },
        confirm_models: PARAM_CONFIRM,
        language: { type: 'string', description: 'Override the ASR language for this run.' },
      },
      required: ['input', 'confirm_models'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          stats: { type: 'object' },
          transcript_preview: { type: 'array', items: { type: 'object' } },
          scenes_preview: { type: 'array', items: { type: 'number' } },
          artifacts: { type: 'object' },
          next: { type: 'string' },
        },
        required: ['stats', 'artifacts', 'next'],
      },
      render: (_args, value) => {
        const lines = []
        const stats = value.stats
        lines.push(
          `扫描完成：时长 ${formatHms(stats.duration)} · 转写 ${stats.chunks} 块（${stats.characters} 字，用时 ${humanDuration(stats.asrSeconds)}）· 场景切点 ${stats.scenes} · 粗筛帧 ${stats.frames} · 总用时 ${humanDuration(stats.elapsedSeconds)}`,
        )
        if (value.transcript_preview.length > 0) {
          lines.push('')
          lines.push('转写预览：')
          for (const entry of value.transcript_preview) lines.push(`- [${entry.start}-${entry.end}] ${entry.text}`)
        }
        if (stats.scenes > 0) {
          lines.push('')
          lines.push(`场景切点前几个（秒）：${value.scenes_preview.join(', ')}`)
        }
        lines.push('')
        lines.push(value.next)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    async execute(args, exec) {
      const { root, settings, env, input } = await prepare(ctx, exec, args.input, config)
      const result = await scan(root, input.path, args, settings, env, { signal: exec.signal })
      const frames = result.frames?.length ?? 0
      return {
        stats: result.stats,
        transcript_preview: (result.transcript.chunks ?? []).slice(0, 5).map((chunk) => ({
          start: chunk.start,
          end: chunk.end,
          text: String(chunk.text ?? '').slice(0, 120),
        })),
        scenes_preview: (result.scenes ?? []).slice(0, 12),
        artifacts: {
          state_dir: result.dir,
          frames_dir: `${result.dir}/frames`,
          transcript: `${result.dir}/transcript.json`,
          scenes: `${result.dir}/scenes.json`,
        },
        next:
          result.plan.mode === 'fast'
            ? `${frames} 张粗筛帧已就绪。快速档跳过帧描述，直接调用 video_local_timeline 整理时间线。`
            : `${frames} 张粗筛帧已就绪。接着调用 video_local_caption 让本地视觉模型逐帧描述；每次调用默认处理 ${settings.captionPerCall} 帧，可重复调用直到 remaining 为 0。`,
      }
    },
  }
}

function captionTool(ctx, config) {
  return {
    name: 'video_local_caption',
    description:
      'Caption sampled frames with the local vision model and give each one an importance score (1-5). It is resumable: every finished frame is stored, and the next call continues where this one stopped, so call it repeatedly until remaining is 0. Defaults to a bounded batch per call so one call does not run for many minutes. Only frames, never the original timeline, are judged here; the leading model should still look at the original frames before concluding. 中文触发词：逐帧描述、画面打分、重要帧、视觉模型看视频。',
    parameters: {
      type: 'object',
      properties: {
        input: PARAM_INPUT,
        limit: { type: 'integer', description: 'Maximum frames to caption in this call (default from settings, usually 60).' },
        vision_model: { type: 'string', description: 'Override the Ollama vision model for this call.' },
        force: { type: 'boolean', description: 'Ignore cached captions and redo the selected frames.' },
      },
      required: ['input'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          model: { type: 'string' },
          total: { type: 'integer' },
          captioned: { type: 'integer' },
          processed: { type: 'integer' },
          remaining: { type: 'integer' },
          elapsed_seconds: { type: 'integer' },
          failures: { type: 'array', items: { type: 'object' } },
          top: { type: 'array', items: { type: 'object' } },
          next: { type: 'string' },
        },
        required: ['model', 'total', 'captioned', 'remaining', 'next'],
      },
      render: (_args, value) => {
        const lines = []
        lines.push(
          `帧描述：${value.model}　已完成 ${value.captioned}/${value.total}（本次 ${value.processed} 帧，${humanDuration(value.elapsed_seconds)}），剩余 ${value.remaining}`,
        )
        if (value.failures.length > 0) {
          lines.push(`失败 ${value.failures.length} 帧：${value.failures.map((failure) => `${failure.at}s ${failure.error}`).join('；')}`)
        }
        if (value.top.length > 0) {
          lines.push('')
          lines.push('当前高分帧：')
          for (const frame of value.top) lines.push(`- ${formatHms(frame.at)}　重要度 ${frame.importance}/5　${frame.caption}`)
        }
        lines.push('')
        lines.push(value.next)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    async execute(args, exec) {
      const { root, settings, env, input } = await prepare(ctx, exec, args.input, config)
      const result = await caption(root, input.path, args, settings, env, { signal: exec.signal })
      return {
        model: result.model,
        total: result.total,
        captioned: result.captioned,
        processed: result.processed,
        remaining: result.remaining,
        elapsed_seconds: result.elapsedSeconds,
        failures: result.failures,
        top: result.top.map((frame) => ({
          at: frame.at,
          caption: frame.caption,
          importance: frame.importance,
          tags: frame.tags ?? [],
        })),
        next:
          result.remaining > 0
            ? `还有 ${result.remaining} 帧没有描述，再调用一次 video_local_caption 继续。`
            : '全部帧已描述，接着调用 video_local_timeline 归纳时间线并挑关键帧。',
      }
    },
  }
}

function timelineTool(ctx, config) {
  return {
    name: 'video_local_timeline',
    description:
      'Fold the timecoded transcript into a timeline: window summaries, chapter headings, keywords and a ranked list of key-frame candidates. Uses the local text model only; the result is screening material, not a conclusion. Requires video_local_scan to have run. Returns the outline, the key-frame list with timecodes and reasons, and writes timeline.md/timeline.json into the state directory. 中文触发词：时间线、纲要、章节、总结视频、关键帧挑选。',
    parameters: {
      type: 'object',
      properties: {
        input: PARAM_INPUT,
        frames: { type: 'integer', description: 'How many key-frame candidates to keep (default from settings, usually 12).' },
        text_model: { type: 'string', description: 'Override the Ollama text model for this call.' },
      },
      required: ['input'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          summary: { type: 'string' },
          keywords: { type: 'array', items: { type: 'string' } },
          outline: { type: 'array', items: { type: 'object' } },
          key_frames: { type: 'array', items: { type: 'object' } },
          coverage: { type: 'object' },
          models: { type: 'object' },
          artifacts: { type: 'object' },
          provenance: { type: 'string' },
          next: { type: 'string' },
        },
        required: ['title', 'outline', 'key_frames', 'artifacts', 'next'],
      },
      render: (_args, value) => {
        const lines = []
        lines.push(`时间线：${value.title}`)
        if (value.keywords.length > 0) lines.push(`关键词：${value.keywords.join('、')}`)
        lines.push('')
        for (const segment of value.outline) {
          lines.push(`### ${clock(segment.start)}–${clock(segment.end)}　${segment.title}`)
          lines.push(segment.summary)
          lines.push('')
        }
        if (value.key_frames.length > 0) {
          lines.push('关键帧候选（主模型应看原始帧再下结论）：')
          for (const frame of value.key_frames) {
            lines.push(`- ${clock(frame.at)}　重要度 ${frame.importance}/5　${frame.caption}　(${frame.reason})`)
          }
          lines.push('')
        }
        if (value.coverage && !value.coverage.complete) {
          lines.push(
            `注意：只描述了 ${value.coverage.captioned}/${value.coverage.frames} 帧，上面的关键帧只覆盖已描述的部分。`,
          )
          lines.push('')
        }
        lines.push(value.provenance)
        lines.push('')
        lines.push(value.next)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    async execute(args, exec) {
      const { root, settings, env, input } = await prepare(ctx, exec, args.input, config)
      const result = await buildTimeline(root, input.path, args, settings, env, { signal: exec.signal })
      const timeline = result.timeline
      return {
        title: timeline.title,
        summary: timeline.outline?.[0]?.summary ?? '',
        keywords: timeline.keywords ?? [],
        outline: (timeline.outline ?? []).map((segment) => ({
          start: segment.start,
          end: segment.end,
          title: segment.title,
          summary: segment.summary,
          headings: segment.headings ?? [],
        })),
        key_frames: (timeline.keyFrames ?? []).map((frame) => ({
          at: frame.at,
          caption: frame.caption,
          importance: frame.importance,
          score: frame.score,
          reason: frame.reason,
        })),
        coverage: timeline.coverage,
        models: timeline.models,
        artifacts: { state_dir: result.dir, timeline_md: `${result.dir}/timeline.md`, timeline_json: `${result.dir}/timeline.json` },
        provenance: timeline.provenance,
        next:
          '需要字幕、Markdown 报告和原始分辨率关键帧时调用 video_local_export；想自己看某一段的原始画面时调用 video_local_frames。',
      }
    },
  }
}

function exportTool(ctx, config) {
  return {
    name: 'video_local_export',
    description:
      'Write the deliverables: subtitles.srt, timeline.md, timeline.json and a report.md under <state>/export/, plus original-resolution JPEGs of the selected key frames under export/frames/. The best frames are also attached to this tool result as images, so the leading model can judge the real pixels instead of the local model caption. 中文触发词：导出字幕、导出关键帧、SRT、报告。',
    parameters: {
      type: 'object',
      properties: {
        input: PARAM_INPUT,
        frames: { type: 'integer', description: 'How many key frames to export and attach (default all kept key frames, capped by the tool).' },
      },
      required: ['input'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          files: { type: 'object' },
          frames: { type: 'array', items: { type: 'object' } },
          images: { type: 'array', items: { type: 'object' } },
          next: { type: 'string' },
        },
        required: ['files', 'frames', 'next'],
      },
      render: (_args, value) => {
        const lines = []
        lines.push('导出完成：')
        for (const [label, path] of Object.entries(value.files)) lines.push(`- ${label}: ${path}`)
        lines.push('')
        if (value.frames.length > 0) {
          lines.push('已导出的关键帧（原始分辨率）：')
          for (const frame of value.frames) lines.push(`- ${clock(frame.at)}　重要度 ${frame.importance}/5　${frame.caption}`)
        }
        const blocks = [{ type: 'text', text: lines.join('\n') }]
        return blocks.concat(imageBlocks(value.images.map((entry) => ({ ref: entry }))))
      },
    },
    async execute(args, exec) {
      const { root, settings, env, input } = await prepare(ctx, exec, args.input, config)
      const result = await exportAll(root, input.path, args, settings, env, { signal: exec.signal })
      const attach = await attachFrames(ctx, result.frames, 6)
      return {
        files: {
          report: result.reportPath,
          timeline_md: result.mdPath,
          subtitles: result.srtPath,
          frames_dir: `${result.outDir}/frames`,
        },
        frames: result.frames.map((frame) => ({
          at: frame.at,
          caption: frame.caption,
          importance: frame.importance,
          path: frame.path,
        })),
        images: attach.map((entry) => entry.ref),
        next:
          '把关键帧和字幕路径交给用户。本地模型的字幕与时间线可能出错，结论请以这些原始帧为准；需要再看某一段时调用 video_local_frames。',
      }
    },
  }
}

function framesTool(ctx, config) {
  return {
    name: 'video_local_frames',
    description:
      'Return original video frames for one time range, so the leading model can see what is actually on screen instead of trusting a local caption. Cheap: it only runs ffmpeg extraction, at most 12 frames per call. Use it after video_local_timeline (or on its own for a specific question) to zoom into a minute or a few seconds around a candidate. 中文触发词：看原始帧、放大某段、取图、视频截图。',
    parameters: {
      type: 'object',
      properties: {
        input: PARAM_INPUT,
        start: { type: 'number', description: 'Range start in seconds.' },
        end: { type: 'number', description: 'Range end in seconds.' },
        count: { type: 'integer', description: 'How many frames to return, 1-12 (default 4).' },
        width: { type: 'integer', description: 'Frame width in pixels (default 1280, never upscales past the source).' },
      },
      required: ['input', 'start', 'end'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          range: { type: 'object' },
          frames: { type: 'array', items: { type: 'object' } },
          images: { type: 'array', items: { type: 'object' } },
        },
        required: ['range', 'frames'],
      },
      render: (_args, value) => {
        const lines = [`原始帧：${clock(value.range.start)}–${clock(value.range.end)}（${value.frames.length} 张）`]
        for (const frame of value.frames) lines.push(`- ${clock(frame.at)} ${frame.path}`)
        const blocks = [{ type: 'text', text: lines.join('\n') }]
        return blocks.concat(imageBlocks(value.images.map((entry) => ({ ref: entry }))))
      },
    },
    async execute(args, exec) {
      const { root, settings, env, input } = await prepare(ctx, exec, args.input, config)
      if (!env.ffmpeg || !env.ffprobe) throw new Error('没有找到 ffmpeg/ffprobe：无法取帧。')
      const result = await grabFrames(root, input.path, args, settings, env, { signal: exec.signal })
      const attach = await attachFrames(ctx, result.frames, 12)
      return {
        range: { start: result.start, end: result.end },
        frames: result.frames.map((frame) => ({ at: frame.at, path: frame.path })),
        images: attach.map((entry) => entry.ref),
      }
    },
  }
}

/* ------------------------------------------------------------- utilities */

function formatHms(seconds) {
  const total = Math.max(0, Number(seconds) || 0)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = Math.floor(total % 60)
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function clock(seconds) {
  return formatHms(seconds)
}

export function apply(ctx, config = {}) {
  ctx.tools.register(planTool(ctx, config))
  ctx.tools.register(scanTool(ctx, config))
  ctx.tools.register(captionTool(ctx, config))
  ctx.tools.register(timelineTool(ctx, config))
  ctx.tools.register(exportTool(ctx, config))
  ctx.tools.register(framesTool(ctx, config))
}
