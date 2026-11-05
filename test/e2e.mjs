/**
 * End-to-end check against one real local video (not shipped as a test fixture).
 *
 *   node test/e2e.mjs "I:\path\to\video.mp4" [frames]
 *
 * It walks the exact functions the tools call, so a failure here is a real
 * pipeline failure rather than a UI quirk. Everything lands under
 * <workspace>/.video/<key>/ and is safe to delete afterwards.
 */
import { resolve } from 'node:path'

import {
  buildTimeline,
  caption,
  detectEnvironment,
  exportAll,
  planFor,
  resolveSettings,
  scan,
} from '../lib/pipeline.js'

const video = process.argv[2]
const frameLimit = Number(process.argv[3] ?? 8)
if (!video) {
  console.error('usage: node test/e2e.mjs <video> [captionFrames]')
  process.exit(2)
}

const root = resolve(process.env.DSH_E2E_ROOT ?? 'I:\\agent10')
const started = Date.now()
const settings = resolveSettings(root, {}).settings

const env = await detectEnvironment(root, settings)
console.log('ffmpeg:', env.ffmpeg)
console.log('python:', env.python, '|', env.pythonDetail)
console.log('asr   :', env.asrModel)
console.log('vision:', env.visionCapabilities.join(', ') || '(none)', '->', env.visionModel)
console.log('text  :', env.textCapabilities.join(', ') || '(none)', '->', env.textModel)

const planned = await planFor(root, video, { mode: 'standard', coarse_sec: 5 }, settings, env)
console.log('plan  :', planned.plan.planId, planned.plan.budget)

const scanned = await scan(root, video, { confirm_models: true, plan_id: planned.plan.planId }, settings, env, {
  onProgress: (event) => console.log('scan  :', JSON.stringify(event)),
})
console.log('scan done', scanned.stats)

const captioned = await caption(root, video, { limit: frameLimit }, settings, env, {
  onProgress: (event) => console.log('cap   :', JSON.stringify(event)),
})
console.log('caption done', {
  total: captioned.total,
  captioned: captioned.captioned,
  remaining: captioned.remaining,
  seconds: captioned.elapsedSeconds,
  failures: captioned.failures,
})

const timeline = await buildTimeline(root, video, {}, settings, env, {
  onProgress: (event) => console.log('tl    :', JSON.stringify(event)),
})
console.log('timeline', {
  title: timeline.timeline.title,
  segments: timeline.timeline.outline.length,
  keyFrames: timeline.timeline.keyFrames.length,
  seconds: timeline.timeline.stats.elapsedSeconds,
})

const exported = await exportAll(root, video, {}, settings, env)
console.log('export', { files: exported.reportPath, frames: exported.frames.length })
console.log('total seconds', Math.round((Date.now() - started) / 1000))
