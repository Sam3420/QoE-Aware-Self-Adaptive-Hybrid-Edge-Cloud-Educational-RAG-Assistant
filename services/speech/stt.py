"""Speech-to-text: SraVaani 1.0 (default; 65 Indian languages and dialects plus English) or distil-whisper (English).

    STT_BACKEND=sravaani   ARTPARK-IISc/SraVaani-1.0: 430M FastConformer, hybrid TDT-CTC, fp16 TorchScript (~900 MB).
                           Gated on Hugging Face (accept its terms once; the HF login token is used) and loaded with
                           trust_remote_code. Meant for a CUDA GPU (fp16); on CPU it is slow.
    STT_BACKEND=whisper    distil-whisper/distil-small.en: English only, transformers pipeline.

Both share `transcribe`: WAV in, text out (in the language spoken, e.g. Devanagari for Hindi), traced as
`transcribe-question` with the model load and the inference as child spans.
"""
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from services.speech.audio import TARGET_RATE, AudioError, decode_wav, encode_wav, resample
from telemetry import tracing

# Only the files the Whisper pipeline needs: the repo also holds ONNX and fp32 copies (> 1.5 GB) that are not used.
STT_FILES = ["*.json", "*.txt", "model.safetensors"]


class Transcript(BaseModel):
    text: str
    duration_s: float  # length of the recording
    latency_ms: float  # transcription time (model load excluded)
    model: str


class SpeechToText:
    """Shared flow: decode, length limit, lazy load, inference, tracing. Subclasses implement _load and _infer."""

    def __init__(self, *, model_id: str, device: str = "auto", cache_dir: Path | None = None,
                 max_seconds: float = 60.0, pipeline: Any = None) -> None:
        self.model_id, self.device, self.cache_dir, self.max_seconds = model_id, device, cache_dir, max_seconds
        self._pipe = pipeline  # tests inject a fake: a callable {"raw", "sampling_rate"} -> {"text"}
        self.device_used: str | None = None
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
            self._pipe = self._load()
            return time.perf_counter() - start

    def _load(self) -> Any:
        raise NotImplementedError

    def _cuda(self) -> bool:
        import torch

        cuda = torch.cuda.is_available() and self.device in ("auto", "cuda")
        self.device_used = "cuda" if cuda else "cpu"
        return cuda

    def transcribe(self, wav: bytes, *, user_id: str | None = None, session_id: str | None = None) -> Transcript:
        """The spoken text ("" for silence). Raises AudioError for an unreadable or too-long recording.

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
                    obs.update(output={"seconds": round(load_s, 1), "device": self.device_used})
            audio = resample(samples, rate, TARGET_RATE)
            with tracing.observation("transcribe-speech", as_type="span", input={"duration_s": round(duration, 2)},
                                     metadata={"model": self.model_id, "device": self.device_used}) as obs:
                start = time.perf_counter()
                text = "" if float(np.abs(audio).max(initial=0.0)) < 1e-3 else self._infer(audio).strip()
                latency = (time.perf_counter() - start) * 1000
                obs.update(output={"text": text, "latency_ms": round(latency)})
            root.update(output={"text": text, "audio_s": round(duration, 2), "latency_ms": round(latency),
                                "real_time_factor": round(latency / 1000 / duration, 3) if duration else None,
                                "model_load_s": round(load_s, 1)})
        return Transcript(text=text, duration_s=round(duration, 2), latency_ms=round(latency, 1), model=self.model_id)

    def _infer(self, audio: np.ndarray) -> str:
        """16 kHz mono float32 -> text. Default: the injected/pipeline callable."""
        return str(self._pipe({"raw": audio, "sampling_rate": TARGET_RATE})["text"])


class WhisperSTT(SpeechToText):
    """distil-whisper/distil-small.en through the transformers ASR pipeline (English only)."""

    def _load(self) -> Any:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        path = snapshot_download(self.model_id, allow_patterns=STT_FILES,
                                 cache_dir=str(self.cache_dir) if self.cache_dir else None)
        cuda = self._cuda()
        return pipeline("automatic-speech-recognition", model=path, device=0 if cuda else -1,
                        dtype=torch.float16 if cuda else torch.float32)


class SraVaaniSTT(SpeechToText):
    """ARTPARK-IISc/SraVaani-1.0 (multilingual Indic + English) with its own transcribe() from trust_remote_code.

    Its documented API takes audio file paths, so each recording is written to a temporary 16 kHz WAV first.
    """

    def __init__(self, *, token: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.token = token

    def _load(self) -> Any:
        from transformers import AutoModel

        cuda = self._cuda()
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, token=self.token,
                                          cache_dir=str(self.cache_dir) if self.cache_dir else None)
        return model.to("cuda" if cuda else "cpu").eval()

    def _infer(self, audio: np.ndarray) -> str:
        if not hasattr(self._pipe, "transcribe"):  # an injected test callable
            return super()._infer(audio)
        import torch

        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encode_wav(audio, TARGET_RATE))
            with torch.inference_mode():
                hyps = self._pipe.transcribe([path], return_hypotheses=True)
        finally:
            os.unlink(path)
        first = hyps[0] if hyps else ""
        return str(getattr(first, "text", first))
