"""Topics offered to learners: NCERT Class 8 Science biology chapters, plus the original sample lesson.

Each topic's lesson text is knowledge_prep/samples/<topic_id>.txt. The Class 8 texts are project-written
lesson notes that follow the NCERT chapter outlines; they are not the textbook's own text. Replace a file with
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
    Topic("crop_production_and_management", "Crop Production and Management", 8, "Biology"),
    Topic("microorganisms_friend_and_foe", "Microorganisms: Friend and Foe", 8, "Biology"),
    Topic("cell_structure_and_functions", "Cell: Structure and Functions", 8, "Biology"),
    Topic("conservation_of_plants_and_animals", "Conservation of Plants and Animals", 8, "Biology"),
    Topic("reproduction_in_animals", "Reproduction in Animals", 8, "Biology"),
    Topic("reaching_the_age_of_adolescence", "Reaching the Age of Adolescence", 8, "Biology"),
    Topic("photosynthesis", "Photosynthesis", 7, "Biology"),  # the project's original sample lesson
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
