"""Speech for accessibility: WAV helpers, STT/TTS services (fake models; nothing is downloaded) and the UI pieces."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from services import speech
from services.speech.audio import AudioError, decode_wav, encode_wav, resample
from services.speech.stt import WhisperSTT
from services.speech.tts import KokoroTTS, speakable


def tone(seconds=1.0, rate=16000, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# --- audio -------------------------------------------------------------------------------------------------------------

def test_wav_round_trip_and_resampling():
    samples, rate = decode_wav(encode_wav(tone(), 16000))
    assert rate == 16000 and len(samples) == 16000 and abs(samples.max() - 0.5) < 1e-3
    assert len(resample(tone(1.0, 24000), 24000, 16000)) == 16000


def test_stereo_and_8_bit_recordings_are_read():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2), w.setsampwidth(1), w.setframerate(8000)
        w.writeframes(bytes([128, 255] * 800))  # left silent, right loud
    samples, rate = decode_wav(buf.getvalue())
    assert rate == 8000 and len(samples) == 800 and 0.4 < samples.mean() < 0.6  # channels averaged


def test_unreadable_recordings_raise_audio_error():
    with pytest.raises(AudioError):
        decode_wav(b"not a wav file")
    with pytest.raises(AudioError, match="empty"):
        decode_wav(encode_wav(np.zeros(0, dtype=np.float32), 16000))


# --- text for speech ---------------------------------------------------------------------------------------------------

def test_speakable_removes_markdown_citations_and_urls():
    text, cut = speakable("**Hint:** think about [the leaf](https://x.y) [1, 2].\n- Step 1: light\n### Done "
                          "https://example.com")
    assert text == "Hint: think about the leaf . Step 1: light Done" and cut is False


def test_long_answers_are_cut_at_a_sentence_boundary():
    text, cut = speakable("First sentence here. " * 10, max_chars=60)
    assert cut and text.endswith(".") and len(text) <= 60


# --- services with fake models -----------------------------------------------------------------------------------------

def test_stt_transcribes_with_the_pipeline_and_skips_silence():
    calls = []

    def pipe(inputs):
        calls.append(inputs)
        return {"text": " What is photosynthesis? "}

    stt = WhisperSTT(model_id="distil-whisper/distil-small.en", pipeline=pipe, max_seconds=5)
    heard = stt.transcribe(encode_wav(tone(1.0, 48000), 48000))  # resampled to 16 kHz
    assert heard.text == "What is photosynthesis?" and heard.duration_s == 1.0
    assert calls[0]["sampling_rate"] == 16000 and len(calls[0]["raw"]) == 16000
    assert stt.transcribe(encode_wav(np.zeros(16000, dtype=np.float32), 16000)).text == ""  # silence
    assert len(calls) == 1
    with pytest.raises(AudioError, match="limit is 5 s"):
        stt.transcribe(encode_wav(tone(6.0), 16000))


def test_tts_returns_playable_wav():
    seen = {}

    def create(text, voice, speed, lang):
        seen.update(text=text, voice=voice, speed=speed, lang=lang)
        return tone(0.5, 24000), 24000

    tts = KokoroTTS(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", model_file="onnx/model_quantized.onnx",
                    engine=NS(create=create))
    out = tts.synthesize("**Plants** make food [1].", voice="bf_emma", speed=1.2)
    samples, rate = decode_wav(out.wav)
    assert rate == 24000 and out.duration_s == 0.5 and out.voice == "bf_emma"
    assert seen == {"text": "Plants make food .", "voice": "bf_emma", "speed": 1.2, "lang": "en-us"}
    with pytest.raises(ValueError, match="no text"):
        tts.synthesize("**[1]**")


def test_status_reports_disabled_features(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "stt_enabled", False)
    assert speech.status()["stt"].startswith("disabled")


# --- UI ----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def fake_speech(monkeypatch):
    """Installed-and-enabled speech with fake models."""
    stt = WhisperSTT(model_id="fake", pipeline=lambda inputs: {"text": "How do plants make food?"})
    tts = KokoroTTS(repo_id="fake", model_file="fake", engine=NS(create=lambda *a, **k: (tone(0.2, 24000), 24000)))
    monkeypatch.setattr(speech, "_stt", stt)
    monkeypatch.setattr(speech, "_tts", tts)
    monkeypatch.setattr(speech, "status", lambda: {"stt": None, "tts": None})


def voice_question_page():
    from types import SimpleNamespace as NS

    import streamlit as st

    from frontend.components import speech as ui
    from services.speech.audio import encode_wav
    from tests.test_speech import tone

    recording = NS(getvalue=lambda: encode_wav(tone(1.0), 16000))
    real = st.chat_input
    st.chat_input = lambda *a, **k: NS(text="", audio=recording)  # the learner tapped the mic and spoke
    try:
        question, spoken = ui.question_box("Ask a doubt", key="ask")
    finally:
        st.chat_input = real
    st.write(f"QUESTION={question}")
    ui.spoken_caption(spoken)
    ui.listen_button("Plants make food using sunlight.", key="a1")


def test_a_voice_question_becomes_the_asked_question_and_answers_can_be_heard(fake_speech):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(voice_question_page, default_timeout=30).run()
    assert not at.exception
    assert any("QUESTION=How do plants make food?" in m.value for m in at.markdown)
    assert any("Asked by voice" in c.value for c in at.caption)
    at = at.button(key="listen-a1").click().run()
    assert not at.exception and len(at.get("audio")) == 1  # the answer is played


def failed_recording_page():
    from types import SimpleNamespace as NS

    import streamlit as st

    from frontend.components import speech as ui

    real = st.chat_input
    st.chat_input = lambda *a, **k: NS(text="", audio=NS(getvalue=lambda: b"garbage"))
    try:
        question, _ = ui.question_box("Ask a doubt", key="ask")
    finally:
        st.chat_input = real
    st.write(f"QUESTION={question}")


def test_an_unreadable_recording_asks_the_learner_to_retry_or_type(fake_speech):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(failed_recording_page, default_timeout=30).run()
    assert not at.exception and "try again or type" in at.error[0].value
    assert any("QUESTION=None" in m.value for m in at.markdown)
