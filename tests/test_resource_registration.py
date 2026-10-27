"""AppGateway.register_resource when a background prefetch registers the same video at the same moment."""
from sqlalchemy import select

from agent.interfaces import FakeMCQProvider, make_memory_gateway
from app.db.models import EducationalResource
from app.db.repositories import LearningTraceRepository


def test_concurrent_insert_of_the_same_video_is_retried_as_an_update(monkeypatch):
    app = make_memory_gateway(retriever=None, mcq_llm=FakeMCQProvider())
    app.ensure_ready()
    original = LearningTraceRepository.create_educational_resource
    raced = []

    def create_after_the_prefetch(self, data):
        if not raced:  # the prefetch's own registration lands between our lookup and our insert
            raced.append(True)
            app.register_resource(source="youtube", external_id=data.external_resource_id, title="from prefetch",
                                  url=None, topic=data.topic, metadata={"n_chunks": 3})
        return original(self, data)

    monkeypatch.setattr(LearningTraceRepository, "create_educational_resource", create_after_the_prefetch)
    resource_id = app.register_resource(source="youtube", external_id="abcdefghijk", title="chosen video",
                                        url="https://www.youtube.com/watch?v=abcdefghijk", topic="heredity",
                                        metadata={"caption_type": "manual"})
    with app.session_factory() as db:
        rows = list(db.scalars(select(EducationalResource)))
    assert len(rows) == 1 and rows[0].id == resource_id and rows[0].title == "chosen video"
    assert rows[0].resource_metadata == {"n_chunks": 3, "caption_type": "manual"}  # both writes kept
