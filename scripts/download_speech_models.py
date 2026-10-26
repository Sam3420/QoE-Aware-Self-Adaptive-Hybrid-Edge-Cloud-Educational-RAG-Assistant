"""Download the speech models from Hugging Face and check them with a round trip (TTS -> STT).

  .venv\\Scripts\\python scripts\\download_speech_models.py            # download + round-trip check
  .venv\\Scripts\\python scripts\\download_speech_models.py --save out  # also keep the synthesised WAV files

STT: distil-whisper/distil-small.en (only model.safetensors + configs, ~335 MB).
TTS: onnx-community/Kokoro-82M-v1.0-ONNX (the ONNX export of hexgrad/Kokoro-82M; int8 model ~92 MB + 5 voices).
The round trip speaks a few lesson sentences with Kokoro, transcribes them with Whisper and reports the word error
rate, so a broken install shows up here instead of in a learner's session.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.speech import get_stt, get_tts, status  # noqa: E402

SENTENCES = [
    "Plants make their own food by photosynthesis using sunlight, water and carbon dioxide.",
    "What is the difference between aerobic and anaerobic respiration?",
    "Dominant traits are expressed even when only one copy of the gene is present.",
]


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (edit distance over reference words)."""
    r, h = words(reference), words(hypothesis)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", type=Path, help="folder to write the synthesised WAV files to")
    args = parser.parse_args()
    problems = {k: v for k, v in status().items() if v}
    if problems:
        print("Speech unavailable:", problems)
        return 1
    stt, tts = get_stt(), get_tts()
    print(f"STT {stt.model_id}: loaded in {stt.load():.1f}s")
    print(f"TTS {tts.repo_id} ({tts.model_file}): loaded in {tts.load():.1f}s")
    errors = []
    for i, sentence in enumerate(SENTENCES, 1):
        speech = tts.synthesize(sentence)
        heard = stt.transcribe(speech.wav)
        errors.append(wer(sentence, heard.text))
        print(f"\n[{i}] said : {sentence}\n    heard: {heard.text}\n    TTS {speech.latency_ms:.0f} ms for "
              f"{speech.duration_s:.1f} s audio · STT {heard.latency_ms:.0f} ms · WER {errors[-1]:.0%}")
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
            (args.save / f"sample_{i}.wav").write_bytes(speech.wav)
    mean = sum(errors) / len(errors)
    print(f"\nRound-trip word error rate: {mean:.1%} ({'OK' if mean <= 0.2 else 'CHECK THE MODELS'})")
    return 0 if mean <= 0.2 else 1


if __name__ == "__main__":
    sys.exit(main())
