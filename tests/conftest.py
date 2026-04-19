"""
Shared fixtures for P5 API tests.
"""
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_p5.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

TEST_ENGINE = create_engine(
    "sqlite:///./test_p5.db",
    connect_args={"check_same_thread": False},
)
TestingSession = sessionmaker(bind=TEST_ENGINE, autoflush=False, autocommit=False)

import api.db.database as _db_mod
_db_mod.engine = TEST_ENGINE
_db_mod.SessionLocal = TestingSession

from api.db.database import Base, get_db
from api.db.models import Prompt, CompletionPair, TrainingRun
from api.main import app


def override_get_db():
    db = TestingSession()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("worker.tasks.launch_training_run", _FakeTask())
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def db():
    Base.metadata.create_all(bind=TEST_ENGINE)
    s = TestingSession()
    try:
        yield s
        s.commit()
    finally:
        s.close()


@pytest.fixture
def sample_prompt(db):
    p = Prompt(id=str(uuid.uuid4()), text="What is the capital of India?", source="test")
    db.add(p)
    db.commit()
    return p


@pytest.fixture
def sample_pair(db, sample_prompt):
    pair = CompletionPair(
        id=str(uuid.uuid4()),
        prompt_id=sample_prompt.id,
        chosen="The capital of India is New Delhi.",
        rejected="India doesn't have a capital.",
        annotator_id="annotator-1",
        confidence=0.95,
    )
    db.add(pair)
    db.commit()
    return pair


class _FakeTask:
    def apply_async(self, **kwargs):
        pass
