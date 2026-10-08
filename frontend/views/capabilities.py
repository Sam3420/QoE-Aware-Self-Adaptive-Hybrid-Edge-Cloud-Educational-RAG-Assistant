"""System capabilities: what the source code implements today, versus the README roadmap."""
import pandas as pd
import streamlit as st

from frontend.components.ui import page_header

# Verified against the source code, not the README. Update when a phase lands.
CAPABILITIES = [
    ("Learner profiles, sessions, interactions (Phase 1)", "Implemented",
     "app/db/models, app/db/repositories/learning_trace.py"),
    ("Text assistant over HTTP (Phase 2)", "Implemented",
     "app/api/routes/interactions.py → AssistantService → Hugging Face"),
    ("Personalization fields (Phase 3)", "Partial",
     "Stored by update_learner_personalization; only preferred language reaches the agent prompt"),
    ("Pre-test / post-test MCQs + feedback", "Implemented",
     "app/services/mcq.py, pretest.py, posttest.py (needs an LLM to generate the bank)"),
    ("YouTube recommendations (Phase 4)", "Not implemented",
     "Only an educational_resources table exists; no YouTube client or API key setting"),
    ("Knowledge prep + FAISS RAG (Phase 5)", "Implemented",
     "knowledge_prep/build_index.py, services/vector_store.py (transcripts are local .txt files)"),
    ("CRAG grading + corrective retrieval (Phase 6)", "Implemented", "services/crag.py, agent/routing.py"),
    ("Speech-to-text (Phase 7)", "Not implemented", "No STT code in the repository"),
    ("Text-to-speech (Phase 8)", "Not implemented", "No TTS code in the repository"),
    ("Monitoring: CPU, memory, RTT (Phase 9)", "Implemented", "services/monitoring.py"),
    ("QoE / SRE / reward evaluation (Phase 9)", "Stub",
     "agent/interfaces.py evaluate_session_stub: rating/5 and a fixed per-route constant"),
    ("Experience memory (Phase 10)", "Implemented", "app/services/experience.py, experiences table"),
    ("Adaptive policy selection (Phase 11)", "Implemented",
     "agent/policy_selection.py: kNN over experiences + ε-exploration (learns from stub rewards)"),
    ("Hybrid local/cloud routing (Phase 12)", "Implemented",
     "services/llm_router.py + agent llm_router node; local = Ollama, cloud = HF, extractive fallback"),
    ("Local model hosting (Phase 13)", "Depends on setup",
     "Uses an Ollama server if one is running with LOCAL_LLM_MODEL pulled"),
]


def render() -> None:
    page_header("System capabilities", "Implementation status",
                "What the current source code does, verified file by file. The README marks Phases 4, 7 and 8 as "
                "implemented; the code does not contain them, so this frontend does not offer them.")
    df = pd.DataFrame(CAPABILITIES, columns=["capability", "status", "evidence"])
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "capability": st.column_config.TextColumn(width="medium"),
        "status": st.column_config.TextColumn(width="small"),
        "evidence": st.column_config.TextColumn(width="large"),
    })
    st.info("Speech input/output and YouTube recommendations are not available in this build.",
            icon=":material/info:")
