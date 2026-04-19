# P5 — RLHF / DPO Alignment Platform

An end-to-end platform for aligning language models with human feedback — covering preference annotation, supervised fine-tuning (SFT), reward model training, and Direct Preference Optimization (DPO).

## Pipeline

```
Prompts + Completions
  └─► Human Annotation UI   (annotators pick preferred response)
        └─► SFT Trainer     (supervised fine-tuning on chosen responses)
              └─► Reward Model  (Bradley-Terry preference model)
                    └─► DPO Trainer   (policy fine-tuning without a RL loop)
                          └─► Serving  (inference endpoint for aligned model)
```

## Features

- **Annotation interface** — side-by-side response comparison with preference capture
- **SFT training** — baseline fine-tuning on high-quality demonstrations
- **Reward model** — learns human preference scores from pairwise data
- **DPO training** — stable, RL-free preference optimization
- **GAE** — Generalized Advantage Estimation for reward shaping experiments
- **Async job queue** — training jobs dispatched via Celery + Redis
- **Model serving** — deploy trained models behind a FastAPI inference endpoint
- **REST API** — manage datasets, annotation tasks, training runs, and deployments

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Task queue | Celery + Redis |
| Database | PostgreSQL (SQLAlchemy) |
| Training | HuggingFace Transformers + PEFT/LoRA |
| DPO | Custom DPO trainer (`trl`-compatible) |
| Serving | FastAPI inference endpoint |

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

- API docs: `http://localhost:8004/docs`
- Annotation UI: `http://localhost:3000`

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/preferences` | Submit an annotator preference |
| `GET` | `/api/v1/preferences` | List collected preferences |
| `POST` | `/api/v1/runs` | Start a training run (SFT/reward/DPO) |
| `GET` | `/api/v1/runs/{id}` | Poll training run status |

## Project Structure

```
├── api/          # REST API — preferences and training run management
├── annotation/   # Annotation workflow and UI backend
├── training/
│   ├── sft_trainer.py      # Supervised fine-tuning
│   ├── reward_model.py     # Preference reward model
│   ├── dpo_trainer.py      # Direct Preference Optimization
│   └── gae.py              # Generalized Advantage Estimation
├── serving/      # Inference endpoint for deployed models
├── worker/       # Celery task definitions for training jobs
├── data/         # Dataset loading and preference data formatting
├── notebooks/    # Exploratory analysis and training experiments
├── scripts/      # Data prep and model export utilities
└── tests/
```

## Training Flow

1. Collect pairwise preferences via annotation UI → stored in `preferences` table
2. Run SFT on chosen responses to get a baseline policy
3. Train a reward model on (chosen, rejected) pairs
4. Run DPO to fine-tune the policy directly from preferences — no RL loop needed
5. Deploy the aligned model via the serving endpoint
