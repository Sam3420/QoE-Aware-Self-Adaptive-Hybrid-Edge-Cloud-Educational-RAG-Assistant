"""Accessibility: ask by voice (mic button in the question box) and listen to answers (text-to-speech).

Typing always works; the mic is offered when speech-to-text is installed and enabled (services/speech). Answers get a
"Listen" button, and the sidebar's Accessibility settings can read new answers aloud automatically.
"""
import hashlib
import inspect
from typing import Any

import streamlit as st

from config import settings
from services import speech
from frontend.utils import state
from services.speech.tts import VOICES

AUTO_KEY, VOICE_KEY, SPEED_KEY, READ_KEY = "tts_auto", "tts_voice", "tts_speed", "tts_already_read"
_CHAT_HAS_AUDIO = "accept_audio" in inspect.signature(st.chat_input).parameters  # Streamlit >= 1.5x


def stt_available() -> bool:
    return _CHAT_HAS_AUDIO and speech.status()["stt"] is None


def tts_available() -> bool:
    return speech.status()["tts"] is None


def question_box(placeholder: str, *, key: str, max_chars: int = 2000, disabled: bool = False,
                 session_id: str | None = None) -> tuple[str | None, dict[str, Any] | None]:
    """The question box: (question, spoken) where spoken describes a transcribed recording (None when typed).

    session_id: the Langfuse session for the speech trace (default: the active agent session).
    """
    if not stt_available():
        return st.chat_input(placeholder, max_chars=max_chars, key=key, disabled=disabled), None
    value = st.chat_input(f"{placeholder} (or tap the mic)", max_chars=max_chars, key=key, disabled=disabled,
                          accept_audio=True, audio_sample_rate=16000)
    if value is None:
        return None, None
    typed = (value.text or "").strip()
    audio = getattr(value, "audio", None)
    if audio is None:
        return typed or None, None
    try:
        with st.spinner("Listening to your question…"):
            ids = _trace_ids()
            heard = speech.get_stt().transcribe(audio.getvalue(), user_id=ids["user_id"],
                                                session_id=session_id or ids["session_id"])
    except Exception as exc:  # unreadable or too long recording, model not downloadable: the learner can type
        st.error(f"Couldn't understand the recording ({exc}). Please try again or type your question.",
                 icon=":material/mic_off:")
        return None, None
    question = " ".join(part for part in (typed, heard.text) if part).strip()
    if not question:
        st.warning("No speech was heard in the recording. Please try again closer to the microphone, or type.",
                   icon=":material/mic_off:")
        return None, None
    return question, {"text": heard.text, "duration_s": heard.duration_s, "latency_ms": heard.latency_ms}


def spoken_caption(spoken: dict[str, Any] | None) -> None:
    if spoken:
        st.caption(f":material/mic: Asked by voice ({spoken['duration_s']:.0f} s, transcribed in "
                   f"{spoken['latency_ms'] / 1000:.1f} s)")


def _trace_ids() -> dict[str, str | None]:
    """The learner and learning session, so the speech traces join that session in Langfuse."""
    return {"user_id": state.get(state.LEARNER), "session_id": state.get(state.AGENT_SESSION)}


@st.cache_data(max_entries=128, show_spinner=False)
def _speech(text: str, voice: str, speed: float, user_id: str | None, session_id: str | None) -> bytes:
    """Cached per text, voice and speed (and session, so a replay elsewhere is traced in its own session)."""
    return speech.get_tts().synthesize(text, voice=voice, speed=speed, user_id=user_id, session_id=session_id).wav


def _prefs() -> tuple[str, float]:
    return st.session_state.get(VOICE_KEY, settings.tts_voice), float(st.session_state.get(SPEED_KEY,
                                                                                         settings.tts_speed))


def listen_button(text: str | None, *, key: str) -> None:
    """A "Listen" button that reads the answer aloud (nothing when TTS is unavailable or there is no text)."""
    if not text or not tts_available():
        return
    if st.button("Listen", icon=":material/volume_up:", key=f"listen-{key}", type="tertiary"):
        _play(text)


def read_aloud_once(text: str | None, *, key: str) -> None:
    """Read a new answer aloud automatically (sidebar setting), once per answer."""
    if not text or not st.session_state.get(AUTO_KEY) or not tts_available():
        return
    done = st.session_state.setdefault(READ_KEY, set())
    if key in done:
        return
    done.add(key)
    _play(text)


def _play(text: str) -> None:
    voice, speed = _prefs()
    try:
        with st.spinner("Preparing audio…"):
            ids = _trace_ids()
            wav = _speech(text, voice, speed, ids["user_id"], ids["session_id"])
    except Exception as exc:
        st.error(f"Couldn't read this aloud ({exc}).", icon=":material/volume_off:")
        return
    st.audio(wav, format="audio/wav", autoplay=True)


def answer_key(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def settings_panel() -> None:
    """Sidebar: accessibility options (shown only when speech is installed)."""
    stt_ok, tts_ok = stt_available(), tts_available()
    with st.expander("Accessibility", icon=":material/accessibility_new:"):
        if stt_ok:
            st.caption(":material/mic: Tap the mic in the question box to ask by voice"
                       + (" in English or an Indian language." if settings.stt_backend == "sravaani" else
                          " (English)."))
        else:
            st.caption(f"Voice questions unavailable: {speech.status()['stt'] or 'needs a newer Streamlit'}.")
        if not tts_ok:
            st.caption(f"Read-aloud unavailable: {speech.status()['tts']}.")
            return
        st.toggle("Read answers aloud automatically", key=AUTO_KEY)
        st.selectbox("Voice", list(dict.fromkeys((settings.tts_voice, *VOICES))), key=VOICE_KEY,
                     format_func=lambda v: {"af": "US female", "am": "US male", "bf": "UK female",
                                            "bm": "UK male"}.get(v[:2], v) + f" · {v[3:].capitalize()}")
        st.slider("Speaking speed", 0.6, 1.4, value=settings.tts_speed, step=0.1, key=SPEED_KEY)
