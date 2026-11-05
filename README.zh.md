# dsh-video-localmodel

> 让 DeepSeek Harness 用**本机的小模型**先把长视频粗筛一遍：本地转写字幕、本地视觉模型看帧、本地文本模型理时间线，最后把**原始候选帧**交给云端大模型做判断。
>
> English: [README.md](README.md)

本地模型的 token 只花时间、不花钱，所以这里可以"多跑几分钟，换云端少看几十帧"。但小模型的输出**只作为初筛和导航**：时间线、纲要、帧描述都可能在细节上出错，最终结论请以导出的原始帧为准。

## 它做什么

```
L0 确定性骨架   ffprobe + 场景切点 + 抽帧           → 秒级，不花模型时间
L1 本地 ASR     Qwen3-ASR-0.6B（静音切分，带时间码） → 约 2.4× 实时（CPU）
L2 本地视觉     qwen3-vl:4b 逐帧描述 + 重要度打分     → 每帧约 3 秒（GPU）
L3 本地文本     qwen2.5:7b 分块归纳 → 章节/纲要/关键词 → 每段约 10 秒
L4 交付         SRT + 时间线 + 原生分辨率关键帧       → 主模型直接看原帧
```

每一步都有缓存（`<工作区>/.video/<文件指纹>/`），中断后重跑会接着上次的进度。

## 工具

| 工具 | 做什么 | 代价 |
|---|---|---|
| `video_local_plan` | 探测 ffmpeg / Python / ASR 权重 / Ollama 模型，报告本次将用哪几个模型和预计耗时 | 不花模型时间 |
| `video_local_scan` | 探针 + 抽音频 + 本地 ASR + 场景切点 + 粗筛帧 | 10 分钟视频约 5 分钟（CPU） |
| `video_local_caption` | 本地视觉模型逐帧描述 + 重要度，可断点续跑 | 每帧约 3 秒（GPU） |
| `video_local_timeline` | 文本模型归纳时间线/章节/纲要，挑关键帧 | 约半分钟 |
| `video_local_export` | 导出 SRT、时间线 Markdown、原始分辨率关键帧，并把最好的几帧作为图片返回 | 十几秒 |
| `video_local_frames` | 取某个时间段的原始帧（最多 12 张） | 秒级 |

**模型确认是硬性的**：`video_local_plan` 会列出这次要用的 ASR / 视觉 / 文本模型；`video_local_scan` 必须带 `plan_id` 和 `confirm_models: true` 才会开始跑本地模型。模型或参数变了，旧 `plan_id` 会被拒绝。

## 快速上手

```
video_local_plan({ input: "D:\\Videos\\lecture.mp4" })
  → 把返回的模型选择和预计耗时给用户看
video_local_scan({ input: "...", plan_id: "<返回的 plan_id>", confirm_models: true })
video_local_caption({ input: "..." })        // 重复调用直到 remaining = 0
video_local_timeline({ input: "..." })
video_local_export({ input: "..." })
```

只想快速看个大概（不跑视觉模型）：`video_local_plan({ input, mode: "fast" })`，然后 scan → timeline → export。

## 依赖

| 依赖 | 必须 | 不装会怎样 |
|---|---|---|
| ffmpeg + ffprobe | 是 | 抽帧、抽音频、场景检测全部不可用 |
| Python（带 `torch` 与 `qwen_asr`） | 是（要字幕的话） | 仍能抽帧和跑画面信号，只是没有台词 |
| Qwen3-ASR 权重目录 | 是（要字幕的话） | 同上 |
| Ollama + 视觉模型（如 `qwen3-vl:4b`） | 标准档 | 退化成"只转写 + 确定性画面信号" |
| Ollama + 文本模型（如 `qwen2.5:7b`） | 否 | 时间线退化成带时间码的原始转写 |

自动探测顺序：插件配置 → `.video/settings.json` → `$DSH_HOME/video-localmodel.json` → `DSH_VIDEO_*` 环境变量 → 常见路径 / `CONDA_EXE` / `PATH`。

## 设置

在 `<工作区>/.video/settings.json` 或 `$DSH_HOME/video-localmodel.json` 写 JSON，键名如下（都可不填）：

| 键 | 默认 | 说明 |
|---|---|---|
| `ffmpegPath` / `ffprobePath` | 自动 | ffmpeg / ffprobe 可执行文件 |
| `pythonPath` | 自动 | 能 `import torch, qwen_asr` 的 Python |
| `asrModelPath` | 自动 | Qwen3-ASR 权重目录 |
| `asrLanguage` | `Chinese` | ASR 语言 |
| `asrThreads` | `0` | ASR 线程数，0 = 库默认 |
| `asrBatch` | `4` | 每批转写的切片数 |
| `asrTargetSec` | `25` | 每块目标时长（静音处切） |
| `ollamaBaseURL` | `http://127.0.0.1:11434` | 与 dsh-plugin-ollama-bridge 共用 |
| `visionModel` | 自动 | Ollama 视觉模型 |
| `textModel` | 自动 | Ollama 文本模型 |
| `coarseSec` | `5` | 粗筛抽帧间隔（秒） |
| `coarseWidth` | `512` | 粗筛帧宽度 |
| `sceneThreshold` | `0.28` | 场景切点阈值 |
| `maxFrames` | `400` | 粗筛帧上限 |
| `maxVideoMinutes` | `30` | 单次视频时长上限 |
| `captionPerCall` | `60` | 每次 `video_local_caption` 最多处理多少帧 |
| `keyFrameWidth` | `1280` | 导出关键帧宽度（不会超过原片） |
| `keepFrames` | `12` | 保留多少关键帧候选 |

## 产物

```
<工作区>/.video/<文件指纹>/
  plan.json  probe.json  scenes.json  transcript.json  frames.json  captions.json
  frames/            # 粗筛帧（512 宽）
  timeline.json      # 时间线 + 关键帧 + 完整转写
  timeline.md        # 给人看的时间线
  export/
    report.md  timeline.md  timeline.json
    subtitles.srt
    frames/          # 原始分辨率关键帧
```

## 实现要点（为什么这样设计）

- **确定性优先**：先跑 ffmpeg 的场景切点和抽帧，再让小模型介入；小模型永远不在"还没有任何结构化信号"的时候被调用。
- **分段可续**：ASR、帧描述、时间线各自落盘。10 分钟视频的帧描述会拆成多次 `video_local_caption`（默认每次 60 帧），每次只有几分钟。
- **思考模型兼容**：`qwen3-vl:4b` 会先输出一段思考链；桥接层给足 `num_predict` 并只取 `content`，思考链不会污染结果。
- **原始帧优先**：交给主模型的是导出的原始分辨率 JPEG，不是本地模型的转述。
- **沙箱友好**：给 Python 子进程单独设置 `TEMP` / `NUMBA_CACHE_DIR` 到工作区，避免受限环境下 numba 在系统临时目录里卡住。

## 已知限制

- 只支持本地文件，暂不支持 B 站 / YouTube 等 URL（下一版）。
- 字幕时间码是"切块级 + 句内按字数分配"的近似值，不是词级对齐；要词级精度需要额外部署 Qwen3-ForcedAligner。
- 画面理解完全取决于本地视觉模型的分辨率与能力；4B 模型在小字、快速切换场景上会漏。
- 没有 GPU 时标准档会非常慢（每帧可能十几秒），建议用 `mode: "fast"` 或只跑 ASR。

## 许可

MIT。本项目不下载、不托管任何视频内容；能不能处理某个文件取决于你是否拥有相应授权。
