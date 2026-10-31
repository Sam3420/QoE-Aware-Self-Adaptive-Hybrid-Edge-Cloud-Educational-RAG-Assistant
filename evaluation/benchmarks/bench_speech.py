"""Result 7: speech accessibility (word error rate and latency of speech-to-text; text-to-speech latency).

    .venv\\Scripts\\python -m evaluation.benchmarks.bench_speech
    .venv\\Scripts\\python -m evaluation.benchmarks.bench_speech --backends sravaani whisper

1. Round trip (synthetic voice): each sentence is spoken by Kokoro (TTS) and transcribed back (STT): the 30 learner
   doubts in English (two voices) and 10 Hindi sentences written for this benchmark (Hindi voice). Synthetic speech
   is cleaner than a real classroom, so these WERs are a lower bound.
2. Real voices (optional, stronger evidence): put WAV files in evaluation/speech_recordings/ with a
   transcripts.csv (file,language,text) and they are transcribed too.

WER = word-level edit distance / reference words, after lower-casing and removing punctuation (Hindi: Devanagari
words, danda removed). Writes evaluation/results/speech/{roundtrip.csv, recordings.csv}.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

from evaluation.benchmarks.isolation import RESULTS, disable_tracing, isolate

isolate("speech", fresh_copy=False)

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from evaluation.benchmarks.harness import load_doubts  # noqa: E402

OUT = RESULTS / "speech"
RECORDINGS = Path(__file__).resolve().parents[1] / "speech_recordings"
HINDI = [  # written for this benchmark: short Class 10 biology questions a learner might ask aloud
    "पौधे सूर्य के प्रकाश से अपना भोजन कैसे बनाते हैं",
    "हमारे शरीर में रक्त का क्या काम है",
    "श्वसन क्रिया में ऊर्जा कैसे निकलती है",
    "मस्तिष्क का कौन सा भाग संतुलन बनाए रखता है",
    "पौधों में हार्मोन क्या करते हैं",
    "बच्चे अपने माता पिता जैसे क्यों दिखते हैं",
    "लड़का या लड़की होना कैसे तय होता है",
    "परागण के बाद क्या होता है",
    "खाद्य श्रृंखला में ऊर्जा कैसे कम होती है",
    "ओज़ोन परत हमारी रक्षा कैसे करती है",
]
EN_VOICES = ("af_heart", "am_michael")


def words(text: str) -> list[str]:
    text = re.sub(r"[।,.?!;:'\"()\-]", " ", text.lower())
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    r, h = words(reference), words(hypothesis)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def make_stt(backend: str):
    from services.speech.stt import SraVaaniSTT, WhisperSTT

    common = {"device": settings.stt_device, "cache_dir": settings.speech_cache_dir,
              "max_seconds": settings.stt_max_seconds}
    if backend == "sravaani":
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        return SraVaaniSTT(model_id=settings.stt_sravaani_model, revision=settings.stt_sravaani_revision, token=token,
                           **common)
    return WhisperSTT(model_id=settings.stt_model, **common)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="+", default=["sravaani", "whisper"], choices=["sravaani", "whisper"])
    args = parser.parse_args()
    disable_tracing()
    OUT.mkdir(parents=True, exist_ok=True)
    from services import speech

    tts = speech.get_tts()
    tts.load()
    sentences = [(d["query"], "en", EN_VOICES[i % 2]) for i, d in enumerate(load_doubts()) if d["kind"] == "doubt"]
    sentences += [(text, "hi", "hf_alpha") for text in HINDI]
    clips = []
    for text, lang, voice in sentences:
        t = time.perf_counter()
        spoken = tts.synthesize(text, voice=voice, lang="hi" if lang == "hi" else "en-us")
        clips.append({"text": text, "language": lang, "voice": voice, "wav": spoken.wav,
                      "audio_s": spoken.duration_s, "tts_ms": (time.perf_counter() - t) * 1000})
    rows = []
    for backend in args.backends:
        stt = make_stt(backend)
        load_s = stt.load()
        for clip in clips:
            if backend == "whisper" and clip["language"] == "hi":
                continue  # distil-whisper is English-only
            heard = stt.transcribe(clip["wav"])
            rows.append({"backend": backend, "device": stt.device_used, "load_s": round(load_s, 2),
                         **{k: v for k, v in clip.items() if k != "wav"}, "heard": heard.text,
                         "stt_ms": heard.latency_ms, "wer": wer(clip["text"], heard.text),
                         "tts_rtf": clip["tts_ms"] / 1000 / max(clip["audio_s"], 1e-6)})
        print(f"{backend}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "roundtrip.csv", index=False)
    print(df.groupby(["backend", "language"]).agg(n=("wer", "size"), wer=("wer", "mean"), stt_ms=("stt_ms", "median"),
                                                  tts_ms=("tts_ms", "median"), audio_s=("audio_s", "mean")))

    manifest = RECORDINGS / "transcripts.csv"
    if manifest.exists():
        rec_rows = []
        for backend in args.backends:
            stt = make_stt(backend)
            for r in csv.DictReader(open(manifest, encoding="utf-8-sig")):
                if backend == "whisper" and r["language"] != "en":
                    continue
                heard = stt.transcribe((RECORDINGS / r["file"]).read_bytes())
                rec_rows.append({"backend": backend, **r, "heard": heard.text, "stt_ms": heard.latency_ms,
                                 "audio_s": heard.duration_s, "wer": wer(r["text"], heard.text)})
        pd.DataFrame(rec_rows).to_csv(OUT / "recordings.csv", index=False)
        print(pd.DataFrame(rec_rows).groupby(["backend", "language"]).wer.agg(["size", "mean"]))
    else:
        print(f"no real recordings ({manifest} missing): only the synthetic round trip was measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
