"""Streamlit session summary shows the measured evaluation (QoE, SRE, reward, predicted vs observed utility)."""
from streamlit.testing.v1 import AppTest


def summary_script() -> None:
    import streamlit as st
    from langgraph.checkpoint.memory import InMemorySaver

    from agent.deps import make_fake_services, set_services
    from agent.graph import build_graph
    from agent.runner import SessionRunner
    from frontend.views import learning_session
    from tests.conftest import ANSWERS, pass_pretest

    if "summary_values" not in st.session_state:
        services = make_fake_services()
        set_services(services)
        runner = SessionRunner(build_graph(InMemorySaver()))
        learner = services.app.create_learner("Summary learner")
        tid = runner.start_session(learner, "photosynthesis")["thread_id"]
        pass_pretest(runner, tid, ANSWERS)
        runner.resume(tid, {"query": "How do plants make glucose from light?"})
        runner.resume(tid, {"end": True})
        final = runner.resume(tid, {"answers": ANSWERS, "rating": 4, "comment": "clear"})["final_state"]
        st.session_state["summary_values"] = (tid, final)
    tid, values = st.session_state["summary_values"]

    class Backend:
        def conversation(self, thread_id):
            return []

    learning_session._summary(Backend(), tid, values)


def test_summary_shows_measured_evaluation():
    at = AppTest.from_function(summary_script, default_timeout=60).run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert {"QoE", "Resource efficiency (SRE)", "Reward", "Utility: observed vs predicted"} <= set(labels)
    captions = " ".join(c.value for c in at.caption)
    assert "configured baseline" in captions  # first session: QoE_old/SRE_old are the baseline
    assert "LQ" in captions and "renormalised" in captions  # unavailable metrics are disclosed
