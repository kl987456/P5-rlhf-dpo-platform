"""
Tests for all P5 API endpoints:
  GET  /health
  POST /api/v1/preferences/prompts
  GET  /api/v1/preferences/prompts
  POST /api/v1/preferences/
  GET  /api/v1/preferences/
  GET  /api/v1/preferences/{id}
  POST /api/v1/runs/
  GET  /api/v1/runs/
  GET  /api/v1/runs/{id}
"""
import uuid

import pytest


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200


# ── POST /preferences/prompts ──────────────────────────────────────────────────

def test_create_prompt(client):
    resp = client.post("/api/v1/preferences/prompts",
                       json={"text": "Explain quantum computing.", "source": "test"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["text"] == "Explain quantum computing."
    assert "id" in body
    assert "created_at" in body


def test_create_prompt_without_source(client):
    resp = client.post("/api/v1/preferences/prompts", json={"text": "Hello?"})
    assert resp.status_code == 201
    assert resp.json()["source"] is None


def test_create_prompt_empty_text_rejected(client):
    resp = client.post("/api/v1/preferences/prompts", json={"text": ""})
    assert resp.status_code == 422


# ── GET /preferences/prompts ───────────────────────────────────────────────────

def test_list_prompts_empty(client):
    resp = client.get("/api/v1/preferences/prompts")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_prompts_returns_created(client, sample_prompt):
    resp = client.get("/api/v1/preferences/prompts")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == sample_prompt.id


def test_list_prompts_pagination(client, db):
    from api.db.models import Prompt
    for i in range(5):
        db.add(Prompt(id=str(uuid.uuid4()), text=f"Prompt {i}"))
    db.commit()

    resp = client.get("/api/v1/preferences/prompts?limit=3&offset=0")
    assert resp.status_code == 200
    assert len(resp.json()) == 3

    resp2 = client.get("/api/v1/preferences/prompts?limit=3&offset=3")
    assert resp2.status_code == 200
    assert len(resp2.json()) == 2


# ── POST /preferences/ ────────────────────────────────────────────────────────

def test_create_preference(client, sample_prompt):
    resp = client.post("/api/v1/preferences/", json={
        "prompt_id": sample_prompt.id,
        "chosen": "Paris is the capital of France.",
        "rejected": "France has no capital.",
        "annotator_id": "user-1",
        "confidence": 0.9,
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["chosen"] == "Paris is the capital of France."
    assert body["rejected"] == "France has no capital."
    assert body["confidence"] == 0.9


def test_create_preference_invalid_prompt(client):
    resp = client.post("/api/v1/preferences/", json={
        "prompt_id": str(uuid.uuid4()),
        "chosen": "A",
        "rejected": "B",
    })
    assert resp.status_code == 404


def test_create_preference_empty_chosen_rejected(client, sample_prompt):
    resp = client.post("/api/v1/preferences/", json={
        "prompt_id": sample_prompt.id,
        "chosen": "",
        "rejected": "B",
    })
    assert resp.status_code == 422


def test_create_preference_confidence_out_of_range(client, sample_prompt):
    resp = client.post("/api/v1/preferences/", json={
        "prompt_id": sample_prompt.id,
        "chosen": "A",
        "rejected": "B",
        "confidence": 1.5,
    })
    assert resp.status_code == 422


# ── GET /preferences/ ─────────────────────────────────────────────────────────

def test_list_preferences_empty(client):
    resp = client.get("/api/v1/preferences/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_preferences_returns_data(client, sample_pair):
    resp = client.get("/api/v1/preferences/")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == sample_pair.id


def test_list_preferences_filter_by_prompt(client, sample_pair, sample_prompt):
    resp = client.get(f"/api/v1/preferences/?prompt_id={sample_prompt.id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp2 = client.get(f"/api/v1/preferences/?prompt_id={uuid.uuid4()}")
    assert resp2.status_code == 200
    assert len(resp2.json()) == 0


# ── GET /preferences/{id} ─────────────────────────────────────────────────────

def test_get_preference(client, sample_pair):
    resp = client.get(f"/api/v1/preferences/{sample_pair.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == sample_pair.id
    assert body["chosen"] == sample_pair.chosen


def test_get_preference_not_found(client):
    resp = client.get(f"/api/v1/preferences/{uuid.uuid4()}")
    assert resp.status_code == 404


# ── POST /runs/ ───────────────────────────────────────────────────────────────

def test_create_sft_run(client):
    resp = client.post("/api/v1/runs/", json={
        "run_type": "sft",
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "config": {"num_epochs": 1, "batch_size": 2},
    })
    assert resp.status_code == 202
    body = resp.json()
    assert body["run_type"] == "sft"
    assert body["status"] == "pending"
    assert "id" in body


def test_create_dpo_run(client):
    resp = client.post("/api/v1/runs/", json={
        "run_type": "dpo",
        "model_name": "sft_weights",
        "config": {"beta": 0.1},
    })
    assert resp.status_code == 202
    assert resp.json()["run_type"] == "dpo"


def test_create_rm_run(client):
    resp = client.post("/api/v1/runs/", json={
        "run_type": "rm",
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "config": {},
    })
    assert resp.status_code == 202
    assert resp.json()["run_type"] == "rm"


def test_create_run_invalid_type(client):
    resp = client.post("/api/v1/runs/", json={
        "run_type": "invalid",
        "model_name": "model",
        "config": {},
    })
    assert resp.status_code == 422


# ── GET /runs/ ────────────────────────────────────────────────────────────────

def test_list_runs_empty(client):
    resp = client.get("/api/v1/runs/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_runs_returns_created(client):
    client.post("/api/v1/runs/", json={"run_type": "sft", "model_name": "m", "config": {}})
    resp = client.get("/api/v1/runs/")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_list_runs_filter_by_type(client):
    client.post("/api/v1/runs/", json={"run_type": "sft", "model_name": "m", "config": {}})
    client.post("/api/v1/runs/", json={"run_type": "dpo", "model_name": "m", "config": {}})

    resp = client.get("/api/v1/runs/?run_type=sft")
    assert resp.status_code == 200
    assert all(r["run_type"] == "sft" for r in resp.json())


# ── GET /runs/{id} ────────────────────────────────────────────────────────────

def test_get_run(client):
    create_resp = client.post("/api/v1/runs/", json={"run_type": "sft", "model_name": "m", "config": {}})
    run_id = create_resp.json()["id"]
    resp = client.get(f"/api/v1/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run_id


def test_get_run_not_found(client):
    resp = client.get(f"/api/v1/runs/{uuid.uuid4()}")
    assert resp.status_code == 404
