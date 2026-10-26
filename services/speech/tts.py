"""Text-to-speech with Kokoro-82M (hexgrad), run through kokoro-onnx on its ONNX export.

The official `kokoro` package pins numpy==1.26.4, which has no Python 3.13 wheel (and would downgrade the project's
numpy), so the model runs from onnx-community/Kokoro-82M-v1.0-ONNX: the same Kokoro-82M v1.0 weights exported to ONNX
(Apache-2.0). kokoro-onnx bundles espeak-ng for phonemisation, so nothing is installed system-wide.
"""
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from services.speech.audio import encode_wav
from telemetry import tracing

VOICES = ("af_heart", "af_bella", "am_michael", "bf_emma", "bm_george")  # downloaded with the model (0.5 MB each)


class Speech(BaseModel):
    wav: bytes
    duration_s: float
    latency_ms: float
    voice: str
    truncated: bool = False


def speakable(text: str, max_chars: int = 1200) -> tuple[str, bool]:
    """Answer text as it should be read aloud: no markdown, citations or URLs, cut at a sentence boundary."""
    text = re.sub(r"\[(\d+(,\s*\d+)*)\]", "", text)  # [1], [2, 3] chunk citations
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>|~]+", "", text)
    text = re.sub(r"^\s*([-•]|\d+[.)])\s+", "", text, flags=re.M)  # list markers
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[:end + 1] if end > max_chars // 2 else cut).strip(), True


class KokoroTTS:
    """Lazy-loaded Kokoro; `synthesize` returns WAV bytes ready for st.audio."""

    def __init__(self, *, repo_id: str, model_file: str, voice: str = "af_heart", speed: float = 1.0,
                 cache_dir: Path | None = None, max_chars: int = 1200, engine: Any = None) -> None:
        self.repo_id, self.model_file, self.voice, self.speed = repo_id, model_file, voice, speed
        self.cache_dir, self.max_chars = cache_dir, max_chars
        self._engine = engine  # tests inject a fake with .create(text, voice=, speed=, lang=)
        self._voices: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> float:
        """Download (first time) and load the model and voices; returns the seconds it took (0 if loaded)."""
        with self._lock:
            if self._engine is not None:
                return 0.0
            start = time.perf_counter()
            from huggingface_hub import hf_hub_download
            from kokoro_onnx import Kokoro

            cache = str(self.cache_dir) if self.cache_dir else None
            model_path = hf_hub_download(self.repo_id, self.model_file, cache_dir=cache)
            wanted = dict.fromkeys((self.voice, *VOICES))
            for name in wanted:  # raw float32 style vectors, (510, 1, 256)
                path = hf_hub_download(self.repo_id, f"voices/{name}.bin", cache_dir=cache)
                self._voices[name] = np.fromfile(path, dtype=np.float32).reshape(-1, 1, 256)
            packed = Path(model_path).parent.parent / "voices.npz"
            if not packed.exists():
                np.savez(packed, **self._voices)
            self._engine = Kokoro(model_path, str(packed))
            return time.perf_counter() - start

    def synthesize(self, text: str, *, voice: str | None = None, speed: float | None = None,
                   user_id: str | None = None, session_id: str | None = None) -> Speech:
        """Speech for an answer (markdown and citations removed; long answers cut at a sentence boundary).

        Traced as `read-aloud` (in the learner's Langfuse session when ids are given), with child spans for the model
        load (first use only) and the synthesis; its output has the latency, audio length and real-time factor.
        """
        name = voice or self.voice
        with tracing.trace("read-aloud", user_id=user_id, session_id=session_id, tags=["speech", "tts"],
                           input={"chars": len(text)}, metadata={"model": self.repo_id, "voice": name},
                           as_type="span") as root:
            spoken, truncated = speakable(text, self.max_chars)
            if not spoken:
                raise ValueError("there is no text to read aloud")
            load_s = 0.0
            if not self.loaded:
                with tracing.observation("load-tts-model", as_type="span",
                                         input={"repo": self.repo_id, "file": self.model_file}) as obs:
                    load_s = self.load()
                    obs.update(output={"seconds": round(load_s, 1)})
            with tracing.observation("synthesize-speech", as_type="span", input=tracing.preview(spoken, 300),
                                     metadata={"model": self.repo_id, "voice": name}) as obs:
                start = time.perf_counter()
                style = self._voices.get(name, name)  # a loaded style vector, or a name the engine knows
                samples, rate = self._engine.create(spoken, voice=style, speed=speed or self.speed, lang="en-us")
                latency = (time.perf_counter() - start) * 1000
                duration = len(samples) / rate
                obs.update(output={"duration_s": round(duration, 2), "latency_ms": round(latency)})
            root.update(output={"chars_spoken": len(spoken), "truncated": truncated, "audio_s": round(duration, 2),
                                "latency_ms": round(latency),
                                "real_time_factor": round(latency / 1000 / duration, 3) if duration else None,
                                "model_load_s": round(load_s, 1)})
        return Speech(wav=encode_wav(samples, rate), duration_s=round(duration, 2), latency_ms=round(latency, 1),
                      voice=name, truncated=truncated)
