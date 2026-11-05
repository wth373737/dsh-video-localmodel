/**
 * Unit smoke test: pure helpers only, so it runs without DSH, ffmpeg or Ollama.
 *   node test/smoke.mjs
 */
import assert from 'node:assert/strict'

import { buildSrt, parseClock, resolveSettings, srtTime } from '../lib/pipeline.js'
import { parseJsonLoose } from '../lib/ollama.js'
import { formatClock, formatTime } from '../lib/media.js'

assert.equal(srtTime(0), '00:00:00,000')
assert.equal(srtTime(65.5), '00:01:05,500')
assert.equal(formatTime(3725), '01:02:05')
assert.equal(formatClock(75), '01:15')
assert.equal(parseClock('02:30'), 150)
assert.equal(parseClock('01:02:03'), 3723)
assert.equal(parseClock('nope'), undefined)

const srt = buildSrt([
  { start: 0, end: 4, text: '第一句话。第二句话。' },
  { start: 4, end: 6, text: '第三句话' },
])
const blocks = srt.trim().split(/\n\n/)
assert.equal(blocks.length, 3, 'one cue per sentence')
assert.match(blocks[0], /^1\n00:00:00,000 --> /)
assert.match(blocks[0], /第一句话。$/)
assert.match(blocks[2], /第三句话$/)

assert.deepEqual(parseJsonLoose('```json\n{"a":1}\n```'), { a: 1 })
assert.deepEqual(parseJsonLoose('前缀 {"a":{"b":2}} 后缀'), { a: { b: 2 } })
assert.equal(parseJsonLoose('not json'), undefined)

const { settings } = resolveSettings(process.cwd(), { coarseSec: 3 })
assert.equal(settings.coarseSec, 3)
assert.equal(settings.visionModel, '')

console.log('smoke ok')
