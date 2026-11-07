# dsh-video-localmodel

> Local-first video screening for DeepSeek Harness: local ASR writes timecoded subtitles, a local vision model captions sampled frames, a local text model folds the transcript into chapters — and the **original candidate frames** go to the cloud model, which makes the final call.
>
> 中文: [README.zh.md](README.zh.md)

Local inference costs time, not money, so this plugin spends minutes locally to save the leading model dozens of frames. The small models are **screening and navigation only**: timelines, outlines and captions can be wrong in the details. The exported original frames are the evidence.

## Pipeline

```
L0 deterministic  ffprobe + scene cuts + frame sampling        seconds, no model time
L1 local ASR      Qwen3-ASR-0.6B, silence-aware chunks         ~2.4x realtime on CPU
L2 local vision   qwen3-vl:4b, caption + importance 1-5        ~3 s per frame on GPU
L3 local text     qwen2.5:7b, window summaries -> chapters     ~10 s per window
L4 hand-off       SRT + timeline + native-resolution frames    leading model reads pixels
```

Every stage writes under `<workspace>/.video/<file-key>/`, so a rerun resumes. ASR is reused while the plan is unchanged; pass `force: true` to `video_local_scan` to re-run it.

## Tools

| Tool | What it does | Cost |
|---|---|---|
| `video_local_plan` | Detect ffmpeg/Python/ASR weights/Ollama models, report the exact models and estimate | no model time |
| `video_local_scan` | Probe, extract audio, local ASR with timecodes, scene cuts, coarse frames | ~5 min per 10 min video (CPU) |
| `video_local_caption` | Local VLM captions + importance, resumable across calls | ~3 s per frame (GPU) |
| `video_local_timeline` | Local text model: timeline, chapters, keywords, key-frame ranking | ~30 s |
| `video_local_export` | SRT, timeline Markdown, native key frames, best frames attached as images | ~15 s |
| `video_local_frames` | Original frames for one time range (max 12) | seconds |

Model choice is explicit: `video_local_plan` reports the ASR/vision/text models, and `video_local_scan` refuses to run without `plan_id` plus `confirm_models: true`. Changing models or parameters invalidates the plan.

## Requirements

- ffmpeg + ffprobe on PATH (or configured)
- Python with `torch` and `qwen_asr`, plus a local Qwen3-ASR checkpoint
- Ollama with a vision model (e.g. `qwen3-vl:4b`) for the standard mode and a text model (e.g. `qwen2.5:7b`)

All paths are configurable; detection order is plugin config → `<workspace>/.video/settings.json` → `$DSH_HOME/video-localmodel.json` → `DSH_VIDEO_*` env vars → conventional locations.

## Output

```
<workspace>/.video/<key>/
  plan.json probe.json scenes.json transcript.json captions.json timeline.json timeline.md
  frames/                 coarse 512px frames
  export/report.md timeline.md timeline.json subtitles.srt frames/  (native key frames)
```

## Notes

- Sentence-level SRT timecodes are approximated by character share inside each ASR chunk; word-level alignment would need Qwen3-ForcedAligner.
- `qwen3-vl` is a thinking model: the bridge budgets extra `num_predict` and keeps only `content`, so reasoning never leaks into captions.
- Python children get `TEMP`/`NUMBA_CACHE_DIR` inside the workspace, which keeps the pipeline alive under a restricted sandbox.
- Local files only in this version; URL ingestion is planned.

## License

MIT. This project ships no video content and provides no way to bypass a site's terms.
