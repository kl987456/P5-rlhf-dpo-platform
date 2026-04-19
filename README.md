# P5 — RLHF / DPO Alignment Platform

End-to-end platform for aligning language models with human feedback — covering preference annotation, SFT, reward model training, and Direct Preference Optimization (DPO) without a RL loop.

## Training pipeline

```
Prompts + model completions
  └─► Human Annotation UI     annotators pick preferred response (A vs B)
        └─► SFT Trainer        QLoRA fine-tuning on chosen responses (baseline policy)
              └─► Reward Model   Bradley-Terry model trained on (chosen, rejected) pairs
                    └─► DPO Trainer   reparameterized RLHF — trains policy directly from preferences
                          └─► ModelServer   FastAPI inference endpoint for the aligned model
```

## Key features

- **SFT with QLoRA** — 4-bit NF4 quantization (bitsandbytes) + LoRA adapters (peft); trains on a single T4 (~4–5 GB VRAM)
- **Reward model** — Bradley-Terry preference model trained on pairwise annotation data
- **DPO** — direct policy optimization from preferences; no reward model needed at inference time
- **GAE** — Generalized Advantage Estimation for reward shaping experiments
- **Celery training queue** — training jobs dispatched asynchronously, one GPU job at a time
- **Inference server** — serves the aligned model behind a FastAPI endpoint
- **SQLite default, PostgreSQL for production**

## Tech stack

| Layer | Package |
|---|---|
| API | fastapi 0.111, uvicorn |
| Task queue | celery 5.3, redis 5 |
| Database | sqlalchemy 2, alembic |
| Training | torch 2.3, transformers 4.41, peft 0.11, bitsandbytes 0.43, datasets 2.19, trl 0.9 |
| Hub push | huggingface_hub 0.23 |
| Tokenization | sentencepiece 0.2 |
| Numerics | numpy 1.26 |

## Project files

```
P5-rlhf-dpo-platform/
├── api/
│   ├── main.py                  FastAPI app with preferences + runs routers
│   ├── security.py              API key auth + rate-limit middleware
│   ├── db/
│   │   ├── database.py          SQLAlchemy engine, session factory
│   │   └── models.py            Prompt, CompletionPair, TrainingRun ORM models
│   ├── models/
│   │   └── schemas.py           Pydantic schemas (PreferenceCreate, TrainingRunCreate, …)
│   └── routes/
│       ├── preferences.py       POST /preferences, GET /preferences, list/filter pairs
│       └── runs.py              POST /runs (trigger SFT/reward/DPO), GET /runs/{id}
├── training/
│   ├── sft_trainer.py           SFT with QLoRA — SFTConfig dataclass, run_sft()
│   ├── reward_model.py          Bradley-Terry reward model — RewardModelConfig, train_reward_model()
│   ├── dpo_trainer.py           DPO loss (Rafailov et al.) — DPORunConfig, run_dpo()
│   └── gae.py                   Generalized Advantage Estimation — compute_gae()
├── serving/
│   └── inference.py             ModelServer — loads checkpoint, exposes /generate endpoint
├── annotation/
│   ├── components/              React annotation UI components (pairwise comparison)
│   └── pages/                   Annotation workflow pages
├── worker/
│   ├── celery_app.py            Celery + Redis configuration
│   └── tasks.py                 Celery tasks for SFT, reward model, DPO training runs
├── data/                        Dataset loading + preference data formatting utilities
├── notebooks/                   Exploratory analysis, reward model evaluation, DPO curves
├── scripts/                     Data prep, model export, checkpoint conversion
├── tests/
│   ├── conftest.py              pytest fixtures (SQLite in-memory, test client)
│   └── test_api.py              API integration tests (preferences CRUD, run lifecycle)
├── docker-compose.yml           redis + api + worker (GPU) + serving (optional profile)
├── Dockerfile                   Multi-stage: base / api / worker / serving targets
├── requirements.txt
├── .env.example
└── .gitignore
```

## DPO loss (implemented in `training/dpo_trainer.py`)

```
L_DPO = -E [ log σ( β · (log π(y_w|x) - log π_ref(y_w|x)
                       - log π(y_l|x) + log π_ref(y_l|x)) ) ]
```

No reward model is needed at inference — the policy itself encodes the preference signal.

## Quick start

```bash
cp .env.example .env          # set HF_TOKEN and BASE_MODEL_NAME
docker compose up --build     # redis + api + worker

# Submit a preference pair via API
curl -X POST http://localhost:8004/api/v1/preferences \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain gravity", "chosen": "...", "rejected": "..."}'

# Trigger a DPO training run
curl -X POST http://localhost:8004/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"run_type": "dpo", "config": {}}'
```

API docs: `http://localhost:8004/docs`

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/preferences` | Submit an annotated preference pair |
| `GET` | `/api/v1/preferences` | List collected preferences |
| `POST` | `/api/v1/runs` | Start a training run (sft / reward / dpo) |
| `GET` | `/api/v1/runs/{id}` | Poll training run status and metrics |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | sqlite:///./rlhf.db | SQLAlchemy connection |
| `CELERY_BROKER_URL` | redis://localhost:6379/0 | Celery broker |
| `BASE_MODEL_NAME` | meta-llama/Llama-3.2-1B-Instruct | Base model for SFT/DPO |
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `OUTPUT_DIR` | ./checkpoints | Local checkpoint save directory |
| `HF_PUSH_REPO` | (blank = skip) | Push trained model to HF Hub |
| `SERVING_PORT` | 8080 | ModelServer port |
| `MAX_NEW_TOKENS` | 512 | Inference max generation length |
