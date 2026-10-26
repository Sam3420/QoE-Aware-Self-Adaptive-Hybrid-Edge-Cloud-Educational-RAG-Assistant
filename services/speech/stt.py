"""Speech-to-text with distil-whisper/distil-small.en (English; transformers pipeline on torch)."""
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from services.speech.audio import TARGET_RATE, AudioError, decode_wav, resample
from telemetry import tracing

# Only the files the pipeline needs: the repo also holds ONNX and fp32 copies (> 1.5 GB) that are not used here.
STT_FILES = ["*.json", "*.txt", "model.safetensors"]


class Transcript(BaseModel):
    text: str
    duration_s: float  # length of the recording
    latency_ms: float  # transcription time (model load excluded)
    model: str


class WhisperSTT:
    """Lazy-loaded ASR pipeline; `transcribe` takes the browser's WAV recording."""

    def __init__(self, *, model_id: str, device: str = "auto", cache_dir: Path | None = None,
                 max_seconds: float = 60.0, pipeline: Any = None) -> None:
        self.model_id, self.device, self.cache_dir, self.max_seconds = model_id, device, cache_dir, max_seconds
        self._pipe = pipeline  # tests inject a fake
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> float:
        """Download (first time) and load the model; returns the seconds it took (0 if already loaded)."""
        with self._lock:
            if self._pipe is not None:
                return 0.0
            start = time.perf_counter()
            import torch
            from huggingface_hub import snapshot_download
            from transformers import pipeline

            path = snapshot_download(self.model_id, allow_patterns=STT_FILES,
                                     cache_dir=str(self.cache_dir) if self.cache_dir else None)
            cuda = torch.cuda.is_available() and self.device in ("auto", "cuda")
            self.device_used = "cuda" if cuda else "cpu"
            self._pipe = pipeline("automatic-speech-recognition", model=path, device=0 if cuda else -1,
                                  dtype=torch.float16 if cuda else torch.float32)
            return time.perf_counter() - start

    def transcribe(self, wav: bytes, *, user_id: str | None = None, session_id: str | None = None) -> Transcript:
        """The spoken English text ("" for silence). Raises AudioError for an unreadable or too-long recording.

        Traced as `transcribe-question` (in the learner's Langfuse session when ids are given), with child spans for
        the model load (first use only) and the inference; its output has the latency and the real-time factor.
        """
        with tracing.trace("transcribe-question", user_id=user_id, session_id=session_id, tags=["speech", "stt"],
                           input={"bytes": len(wav)}, metadata={"model": self.model_id}, as_type="span") as root:
            samples, rate = decode_wav(wav)
            duration = len(samples) / rate
            if duration > self.max_seconds:
                raise AudioError(f"the recording is {duration:.0f} s long; the limit is {self.max_seconds:.0f} s")
            load_s = 0.0
            if not self.loaded:
                with tracing.observation("load-stt-model", as_type="span", input={"model": self.model_id}) as obs:
                    load_s = self.load()
                    obs.update(output={"seconds": round(load_s, 1), "device": getattr(self, "device_used", None)})
            audio = resample(samples, rate, TARGET_RATE)
            with tracing.observation("transcribe-speech", as_type="span", input={"duration_s": round(duration, 2)},
                                     metadata={"model": self.model_id,
                                               "device": getattr(self, "device_used", None)}) as obs:
                start = time.perf_counter()
                text = "" if float(np.abs(audio).max(initial=0.0)) < 1e-3 else str(
                    self._pipe({"raw": audio, "sampling_rate": TARGET_RATE})["text"]).strip()
                latency = (time.perf_counter() - start) * 1000
                obs.update(output={"text": text, "latency_ms": round(latency)})
            root.update(output={"text": text, "audio_s": round(duration, 2), "latency_ms": round(latency),
                                "real_time_factor": round(latency / 1000 / duration, 3) if duration else None,
                                "model_load_s": round(load_s, 1)})
        return Transcript(text=text, duration_s=round(duration, 2), latency_ms=round(latency, 1), model=self.model_id)
