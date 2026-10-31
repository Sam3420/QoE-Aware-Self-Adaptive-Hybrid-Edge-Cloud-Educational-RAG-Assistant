"""Speech for accessibility: speech-to-text for spoken questions and text-to-speech for reading answers aloud.

    STT: SraVaani 1.0 (Indian languages + English) or distil-whisper   services/speech/stt.py
    TTS: Kokoro-82M, ONNX export of hexgrad/Kokoro-82M (kokoro-onnx)     services/speech/tts.py

Both are optional (requirements-speech.txt) and load lazily on first use; models come from Hugging Face into the
normal HF cache (or SPEECH_CACHE_DIR). Pre-download and check with: python scripts/download_speech_models.py
"""
import threading
from typing import Any

from config import settings

_lock = threading.Lock()
_stt: Any = None
_tts: Any = None


def get_stt() -> Any:
    """The shared speech-to-text model for STT_BACKEND (it loads on the first transcription)."""
    global _stt
    with _lock:
        if _stt is None:
            from services.speech.stt import SraVaaniSTT, WhisperSTT

            common = {"device": settings.stt_device, "cache_dir": settings.speech_cache_dir,
                      "max_seconds": settings.stt_max_seconds}
            if settings.stt_backend == "sravaani":
                token = settings.hf_token.get_secret_value() if settings.hf_token else None
                _stt = SraVaaniSTT(model_id=settings.stt_sravaani_model, revision=settings.stt_sravaani_revision,
                                   token=token, **common)
            else:
                _stt = WhisperSTT(model_id=settings.stt_model, **common)
        return _stt


def get_tts() -> Any:
    """The shared KokoroTTS (model loads on first synthesis)."""
    global _tts
    with _lock:
        if _tts is None:
            from services.speech.tts import KokoroTTS

            _tts = KokoroTTS(repo_id=settings.tts_repo, model_file=settings.tts_model_file,
                             voice=settings.tts_voice, speed=settings.tts_speed, cache_dir=settings.speech_cache_dir,
                             max_chars=settings.tts_max_chars, device=settings.tts_device,
                             gpu_model_file=settings.tts_gpu_model_file)
        return _tts


def status() -> dict[str, str | None]:
    """Why STT/TTS is unavailable (None when it can be used): disabled, or its packages are not installed."""
    def missing(*modules: str) -> str | None:
        import importlib.util

        absent = [m for m in modules if importlib.util.find_spec(m) is None]
        return f"not installed ({', '.join(absent)}): pip install -r requirements-speech.txt" if absent else None

    return {"stt": "disabled (STT_ENABLED=false)" if not settings.stt_enabled else missing("torch", "transformers"),
            "tts": "disabled (TTS_ENABLED=false)" if not settings.tts_enabled else missing("kokoro_onnx", "onnxruntime")}
