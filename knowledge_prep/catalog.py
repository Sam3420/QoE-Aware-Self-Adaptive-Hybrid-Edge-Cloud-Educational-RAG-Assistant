"""Topics offered to learners: the biology chapters of NCERT Class 10 Science (rationalised syllabus).

Each topic's lesson text is knowledge_prep/samples/<topic_id>.txt. The texts are project-written lesson
notes that follow the NCERT chapter outlines; they are not the textbook's own text. Replace a file with
the official chapter text or a lecture transcript (then rebuild its index) to change what the assistant
retrieves and what the quizzes are generated from.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    topic_id: str
    title: str
    grade: int
    subject: str


TOPICS: tuple[Topic, ...] = (
    Topic("life_processes", "Life Processes", 10, "Biology"),
    Topic("control_and_coordination", "Control and Coordination", 10, "Biology"),
    Topic("how_do_organisms_reproduce", "How do Organisms Reproduce?", 10, "Biology"),
    Topic("heredity", "Heredity", 10, "Biology"),
    Topic("our_environment", "Our Environment", 10, "Biology"),
)

_BY_ID = {topic.topic_id: topic for topic in TOPICS}


def get_topic(topic_id: str) -> Topic | None:
    return _BY_ID.get(topic_id)


def topic_title(topic_id: str) -> str:
    """Display title; topics outside the catalogue (e.g. uploaded transcripts) use their id."""
    topic = _BY_ID.get(topic_id)
    return topic.title if topic else topic_id.replace("_", " ").replace("-", " ").capitalize()


def topic_label(topic_id: str) -> str:
    """Title with class and subject, for pickers."""
    topic = _BY_ID.get(topic_id)
    if topic:
        return f"Class {topic.grade} {topic.subject} · {topic.title}"
    if topic_id.startswith("yt_"):  # a video transcript index built by a session
        return f"YouTube video · {topic_id[3:]}"
    return f"Uploaded · {topic_title(topic_id)}"


def search_text(topic_id: str) -> str:
    """Words to search external sources (e.g. YouTube) with: title plus class for catalogue topics."""
    topic = _BY_ID.get(topic_id)
    return f"{topic.title} class {topic.grade}" if topic else topic_title(topic_id)


def catalog_order(topic_id: str) -> tuple[int, int | str]:
    """Sort key: catalogue topics in catalogue order, then anything else alphabetically."""
    ids = list(_BY_ID)
    return (0, ids.index(topic_id)) if topic_id in _BY_ID else (1, topic_id)
