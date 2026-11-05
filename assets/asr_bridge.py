#!/usr/bin/env python
"""Qwen3-ASR bridge for dsh-video-localmodel.

Reads one 16 kHz mono PCM16 WAV, splits it at quiet points into ~25 s chunks and
transcribes every chunk with the local Qwen3-ASR checkpoint, so every returned
line carries a start/end timecode. Progress lines go to stderr as
``PROGRESS {json}``; the final result is written to ``--out`` as UTF-8 JSON.

The audio is read with the standard library and handed to the model as an
in-memory (ndarray, sample_rate) pair, which avoids the librosa/audioread path
(and its temporary-file behaviour) entirely.
"""

import argparse
import json
import os
import sys
import time
import wave

import numpy as np


def log(message):
    print(message, file=sys.stderr, flush=True)


def progress(payload):
    print("PROGRESS " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def read_wav(path):
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if width != 2:
        raise SystemExit(f"expected 16-bit PCM, got {width * 8}-bit: {path}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


def split_points(audio, rate, target_sec, min_sec, lookback_sec=6.0):
    """Cut after ~target seconds at the quietest window in the trailing lookback."""
    frame = max(1, int(0.05 * rate))
    limit = len(audio)
    target = int(target_sec * rate)
    floor = int(min_sec * rate)
    lookback = int(lookback_sec * rate)
    cuts = [0]
    start = 0
    while start < limit:
        end = min(limit, start + target)
        if end < limit:
            low = max(start + floor, end - lookback)
            segment = audio[low:end]
            count = len(segment) // frame
            if count > 0:
                rms = np.sqrt((segment[: count * frame].reshape(count, frame) ** 2).mean(axis=1))
                end = max(low + int(np.argmin(rms)) * frame, start + floor)
        cuts.append(end)
        start = end
    return cuts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--target-sec", type=float, default=25.0)
    parser.add_argument("--min-sec", type=float, default=12.0)
    parser.add_argument("--max-tokens", type=int, default=448)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    # numba/librosa caches inside TEMP; the caller points TEMP at a workspace dir.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    started = time.time()
    import torch  # imported late so a bad env fails with a clear message

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    audio, rate = read_wav(args.wav)
    duration = len(audio) / float(rate)
    if duration <= 0.05:
        raise SystemExit(f"audio is empty: {args.wav}")

    cuts = split_points(audio, rate, args.target_sec, args.min_sec)
    spans = [(cuts[i] / rate, cuts[i + 1] / rate) for i in range(len(cuts) - 1)]
    progress({"stage": "split", "chunks": len(spans), "audioSeconds": round(duration, 2)})

    from qwen_asr import Qwen3ASRModel

    load_started = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map=args.device,
        max_inference_batch_size=max(1, args.batch),
        max_new_tokens=args.max_tokens,
    )
    load_seconds = time.time() - load_started
    progress({"stage": "model", "seconds": round(load_seconds, 1)})

    chunks = []
    batch = max(1, args.batch)
    total = len(spans)
    for begin in range(0, total, batch):
        group = list(range(begin, min(total, begin + batch)))
        payload = [(audio[int(spans[i][0] * rate): int(spans[i][1] * rate)], rate) for i in group]
        chunk_started = time.time()
        try:
            results = model.transcribe(audio=payload, language=[args.language] * len(payload))
        except Exception as error:  # a batch failure must not lose the whole run
            log(f"batch failed ({error}); retrying one by one")
            results = []
            for item in payload:
                results.append(model.transcribe(audio=item, language=args.language)[0])
        elapsed = time.time() - chunk_started
        for offset, result in enumerate(results):
            index = group[offset]
            start, end = spans[index]
            chunks.append(
                {
                    "index": index,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": str(getattr(result, "text", "") or "").strip(),
                }
            )
        progress(
            {
                "stage": "asr",
                "done": len(chunks),
                "total": total,
                "seconds": round(elapsed, 1),
                "cursor": chunks[-1]["end"] if chunks else 0,
            }
        )

    payload = {
        "model": args.model,
        "language": args.language,
        "audioSeconds": round(duration, 2),
        "elapsedSeconds": round(time.time() - started, 1),
        "loadSeconds": round(load_seconds, 1),
        "chunks": chunks,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    progress({"stage": "done", "seconds": round(time.time() - started, 1)})


if __name__ == "__main__":
    main()
