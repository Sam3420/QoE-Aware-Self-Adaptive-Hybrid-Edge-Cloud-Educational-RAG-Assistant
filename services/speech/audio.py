"""WAV in and out with the standard library and numpy (no ffmpeg): decode the browser's recording, encode speech."""
import io
import wave

import numpy as np

TARGET_RATE = 16_000  # what Whisper expects (Streamlit's chat_input records at this rate by default)


class AudioError(ValueError):
    """The audio could not be read (not a PCM WAV file, or empty)."""


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 samples in [-1, 1] and the sample rate of a PCM WAV file."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"not a PCM WAV recording: {exc}") from exc
    if not raw:
        raise AudioError("the recording is empty")
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise AudioError(f"unsupported sample width {width * 8} bit")
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    samples = (samples - 128.0) / 128.0 if width == 1 else samples / float(np.iinfo(dtype).max)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def resample(samples: np.ndarray, rate: int, target: int = TARGET_RATE) -> np.ndarray:
    """Linear-interpolation resampling (enough for speech recognition input)."""
    if rate == target or len(samples) == 0:
        return samples.astype(np.float32)
    n = int(round(len(samples) * target / rate))
    return np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    """16-bit PCM mono WAV bytes (what st.audio plays)."""
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
