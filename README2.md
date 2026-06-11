# Psychological State Memory (PSM)

A lightweight psychological memory system for conversational agents that compresses long conversation histories into a persistent Big Five personality state.

## Architecture

```
psm/
├── psm/                        # Core library
│   ├── __init__.py
│   ├── state.py                # PersonalityState dataclass
│   ├── database.py             # SQLite persistence
│   ├── personality_estimator.py# Big Five HuggingFace estimator
│   ├── renderer.py             # Big Five → natural language
│   ├── graph.py                # LangGraph workflow
│   └── nodes/                  # LangGraph nodes
│       ├── __init__.py
│       ├── conversation.py
│       ├── memory_trigger.py
│       ├── personality_estimator.py
│       ├── state_update.py
│       ├── personality_renderer.py
│       └── generation.py
├── ui/
│   └── app.py                  # Streamlit UI
├── experiments/                # Optional experiments
│   ├── exp1_behavior/
│   ├── exp2_efficiency/
│   └── exp3_adaptation/
├── case_studies/
├── tests/
├── config.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download the Qwen3-8B GGUF model and set the path in `config.py` or via environment variable:
```bash
export PSM_MODEL_PATH=/path/to/qwen3-8b.gguf
```

## Run

```bash
streamlit run ui/app.py
```

## Stack

| Component | Library |
|---|---|
| LLM | Qwen3-8B-Instruct (GGUF via llama.cpp) |
| Workflow | LangGraph |
| UI | Streamlit |
| Storage | SQLite |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Big Five | Pretrained HuggingFace predictor |
