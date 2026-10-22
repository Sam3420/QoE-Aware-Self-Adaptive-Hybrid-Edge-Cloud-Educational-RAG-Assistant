"""Matching learner questions and videos to a topic's concepts (contract only; implemented on day 2).

Concept ids are those of knowledge_prep/concepts/<topic_id>.json (see tracker.get_concepts).
"""


def match_question(text: str, topic_id: str) -> tuple[str, float]:
    """The concept a learner's question is about, and the confidence in [0, 1].

    Args:
        text: the learner's question as typed.
        topic_id: catalogue topic id; its vocabulary must exist.

    Returns:
        (concept_id, confidence). The caller decides whether the confidence is high enough to act on (for example to
        call ``tracker.mark_covered(..., "asked")``).

    Raises:
        tracker.VocabularyNotFoundError: the topic has no vocabulary.
    """
    raise NotImplementedError


def video_coverage(video_id: str, topic_id: str) -> list[str]:
    """Concept ids a YouTube video covers, judged from its transcript (empty if none, or no transcript).

    Args:
        video_id: YouTube video id (the transcript index is ``yt_<video_id>``).
        topic_id: catalogue topic id; its vocabulary must exist.

    Returns:
        Concept ids in vocabulary order. The caller records them with ``tracker.mark_covered(..., "video")``.

    Raises:
        tracker.VocabularyNotFoundError: the topic has no vocabulary.
    """
    raise NotImplementedError
