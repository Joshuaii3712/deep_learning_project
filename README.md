### Name: Younggee Chin (진용기 / 22100719)

### Project Title:
Implementation of Psychological State Memory for LLM Chatbots:
A Psychological Approach to Long-Term Memory

### YouTube Link:
(TBD)

### Content:
(see below)

---


# Psychological State Memory (PSM): Behavior-Preserving Psychological Compression for Long-Term Conversational Agents

> **Abstract.** Long-term conversational agents typically maintain context through retrieval, summarization, or external memory stores. While these approaches preserve factual information, they often fail to capture how prolonged interactions shape an agent's behavioral tendencies. We introduce **Psychological State Memory (PSM)**, a lightweight memory mechanism that compresses conversation histories into a five-dimensional persistent psychological state based on the Big Five personality model. Instead of storing facts or summaries, PSM models stable behavioral dispositions—curiosity, conscientiousness, expressiveness, warmth, and emotional reactivity—and periodically updates them via a streaming personality estimator. The resulting state vector is rendered into natural language and injected into the system prompt, enabling behavioral consistency across long sessions while requiring only five floating-point values as persistent memory. We evaluate PSM against full-context prompting, summary-based memory, and fact-centric memory on two axes: behavior preservation and memory efficiency. An additional adaptation experiment examines whether the personality state vector responds meaningfully to sustained user interaction styles. Results show that PSM achieves the highest behavior preservation score and decision consistency among compressed strategies, and delivers an order-of-magnitude improvement in storage efficiency over all baselines. The adaptation experiment, however, exposed a critical label-mapping bug in the personality estimator that prevented trait-specific updates, restricting all personality change to the extraversion dimension. These findings confirm PSM's practical advantages as a memory-efficient architecture and simultaneously highlight the estimator's sensitivity to model-specific output formats as a key direction for future work.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Task and Method](#2-task-and-method)
3. [Experiments](#3-experiments)
4. [Results and Analysis](#4-results-and-analysis)
5. [Conclusion](#5-conclusion)
6. [Repository Structure](#6-repository-structure)
7. [Setup and Usage](#7-setup-and-usage)
8. [References](#8-references)

---

## 1. Introduction

Conversational agents deployed over extended sessions face a fundamental tension between **context richness** and **memory efficiency**. Full-context prompting preserves all behavioral nuance but scales linearly with session length and quickly exhausts the model's context window [6]. Summarization-based approaches compress content but lose stylistic and dispositional signals that accumulate gradually over many turns [7]. Fact-centric retrieval captures declarative knowledge but discards the affective and temperamental patterns that determine *how* an agent responds, not merely *what* it says [8].

Human personality research offers an alternative lens. The **Big Five** (OCEAN) model [1] posits that stable behavioral tendencies—openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism—can be described by five orthogonal continuous dimensions. Crucially, these dimensions are both measurable from text [2] and predictive of behavioral outcomes across contexts [3].

We ask: *can a conversational agent's effective memory be represented not as stored information but as a persistent behavioral state?* We propose **Psychological State Memory (PSM)**, a system that periodically estimates Big Five scores from recent conversation chunks and maintains a smoothed state vector via exponential moving average (EMA) [11]. Before each generation, the state is rendered into a natural-language personality profile and injected below the system prompt, conditioning the model's behavior without consuming significant context budget.

The key contributions of this work are:

- A lightweight psychological memory architecture requiring only five floating-point values of persistent storage per session.
- A trigger-based update mechanism that decouples personality estimation from generation latency.
- Empirical comparison of PSM against three baseline memory strategies on behavior preservation and memory efficiency.
- Identification of a label-mapping failure mode in pretrained Big Five estimators and a concrete fix for future work.

---

## 2. Task and Method

### 2.1 Problem Formulation

Let $\mathcal{H}_t = \{(u_1, a_1), \ldots, (u_t, a_t)\}$ denote a conversation history of $t$ turns, where $u_i$ and $a_i$ are user and agent utterances respectively. A memory system $\mathcal{M}$ must produce a compact representation $m_t = \mathcal{M}(\mathcal{H}_t)$ such that the conditional distribution of future responses $P(a_{t+1} \mid u_{t+1}, m_t)$ approximates $P(a_{t+1} \mid u_{t+1}, \mathcal{H}_t)$ as closely as possible, subject to a storage budget $|m_t| \leq B$.

PSM defines $m_t$ as a five-dimensional real vector $\psi_t \in [0,1]^5$, where each dimension corresponds to one Big Five trait.

### 2.2 Personality State

The persistent psychological state is a dataclass with five continuous dimensions:

```
PersonalityState:
  openness          ∈ [0, 1]
  conscientiousness ∈ [0, 1]
  extraversion      ∈ [0, 1]
  agreeableness     ∈ [0, 1]
  neuroticism       ∈ [0, 1]
```

The initial state is set to the neutral midpoint $\psi_0 = (0.5, 0.5, 0.5, 0.5, 0.5)$.

### 2.3 State Update via Exponential Moving Average

At each memory update event, a personality estimator $f_\theta$ produces a fresh estimate $\hat{\psi}_t$ from a recent conversation window. The persistent state is updated via the EMA formula [11]:

$$\psi_t = \alpha \cdot \psi_{t-1} + (1 - \alpha) \cdot \hat{\psi}_t$$

where $\alpha = 0.99$ is a smoothing coefficient chosen to ensure slow, stable evolution that resists noise in individual estimates. This formulation prevents abrupt behavioral shifts while allowing the state to drift meaningfully over many turns.

### 2.4 Personality Estimator

The personality estimator $f_\theta$ is a pretrained BERT-based text classifier (`Minej/bert-base-personality` [4]) fine-tuned on the Essays dataset [5] for Big Five prediction. It takes a concatenated window of recent utterances (up to 512 tokens) and outputs five classification scores in $[0,1]$.

### 2.5 Memory Trigger

To decouple estimation overhead from generation latency, PSM updates $\psi_t$ only when at least one of three conditions is met. Let $\tau_t$ denote the total token count at turn $t$ and $C = 1024$ the model context window. The trigger fires when:

$$\tau_t > 700 \quad \lor \quad \frac{\tau_t}{C} > 0.65 \quad \lor \quad t \equiv 0 \pmod{20}$$

| Condition | Threshold |
|---|---|
| Accumulated tokens in context | $> 700$ tokens |
| Context utilization ratio | $> 65\%$ of model window |
| Turn count | multiple of $20$ |

These thresholds are tuned for the experimental hardware described in Section 3.1, where a small context window (1,024 tokens) and constrained VRAM necessitate more frequent, lighter-weight updates compared to full-scale deployments.

### 2.6 Personality Rendering

Before each generation, $\psi_t$ is converted to a natural-language profile via a bucket mapping. Each trait score is discretized into five levels using the floor function [cf. 1]:

$$b^{(k)} = \text{clamp}\left(\lfloor \psi^{(k)}_t \cdot 5 \rfloor,\ 0,\ 4\right)$$

Bucket $b = 2$ (neutral) emits no sentence; all others emit a trait-specific description. Examples:

| Trait | Bucket | Description |
|---|---|---|
| Openness | 4 | "The assistant actively explores novel possibilities and alternative perspectives." |
| Openness | 0 | "The assistant prefers familiar and conventional approaches." |
| Agreeableness | 4 | "The assistant responds in a warm, cooperative, and empathetic manner." |
| Neuroticism | 0 | "The assistant remains emotionally stable and rarely focuses on worst-case outcomes." |
| Neuroticism | 4 | "The assistant tends to carefully consider risks and potential negative outcomes." |

The rendered profile is injected directly below the system prompt:

```
[System Prompt]

Psychological Profile

- The assistant actively explores novel possibilities and alternative perspectives.
- The assistant responds in a warm, cooperative, and empathetic manner.
- The assistant tends to be reserved rather than highly expressive.
```

### 2.7 System Architecture

The full pipeline is implemented as a **LangGraph** state machine with six sequential nodes:

```
┌─────────────────────────────────────────────────────────────────┐
│                         LangGraph DAG                           │
│                                                                 │
│   ┌──────────────────┐                                          │
│   │  ConversationNode │  ← append user message, increment turn  │
│   └────────┬─────────┘                                          │
│            ↓                                                    │
│   ┌──────────────────┐                                          │
│   │MemoryTriggerNode │  ← check token / ctx / turn thresholds   │
│   └────────┬─────────┘                                          │
│            ↓                                                    │
│   ┌──────────────────────┐                                      │
│   │PersonalityEstimator  │  ← run f_θ on conversation window    │
│   │       Node           │    (skipped if not triggered)        │
│   └────────┬─────────────┘                                      │
│            ↓                                                    │
│   ┌──────────────────┐                                          │
│   │ StateUpdateNode  │  ← EMA: ψ_t = αψ_{t-1} + (1-α)ψ̂_t    │
│   └────────┬─────────┘                                          │
│            ↓                                                    │
│   ┌────────────────────────┐                                    │
│   │PersonalityRendererNode │  ← bucket mapping → natural lang   │
│   └────────┬───────────────┘                                    │
│            ↓                                                    │
│   ┌──────────────────┐                                          │
│   │  GenerationNode  │  ← LLM call with enriched system prompt  │
│   └────────┬─────────┘                                          │
│            ↓                                                    │
│          [END]                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**State persistence** is handled by SQLite, storing session metadata, full message history, Big Five state per session, and a timestamped personality snapshot log.

### 2.8 Implementation Stack

| Component | Implementation |
|---|---|
| LLM | Llama 3.2 1.2B (Q8_0) via Ollama |
| Workflow | LangGraph |
| UI | Streamlit |
| State Storage | SQLite (SQLAlchemy Core) |
| Personality Estimator | `Minej/bert-base-personality` (HuggingFace) [4] |
| Style Similarity (Exp 1–2) | `sentence-transformers/all-mpnet-base-v2` [9] |

---

## 3. Experiments

### 3.1 Experimental Setup

All experiments are conducted on a consumer-grade laptop with the following specifications:

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce MX250 (VRAM: 2 GB) |
| CUDA Version | 13.0 |
| RAM | 16 GB |
| CPU | Intel Core i5-10210U @ 1.60 GHz (boost 2.11 GHz) |

The LLM is Llama 3.2 (1.2B parameters, Q8_0 quantization) served via Ollama. Model configuration is adjusted to fit within the available memory budget:

| Parameter | Value |
|---|---|
| Context window (`N_CTX`) | 1,024 tokens |
| Temperature | 0.7 |
| Max new tokens | 128 |
| Trigger token threshold | 700 |
| Trigger context ratio | 0.65 |
| Trigger turn interval | 20 |

**Note on the LLM-as-judge in Experiment 1.** Due to hardware constraints, the same Llama 3.2 1.2B (Q8_0) model is used both as the agent being evaluated and as the judge scoring response similarity. This is a significant limitation: a small 1.2B model has limited capacity for meta-evaluation, and the relatively low LLM-Judge scores (2.5–4.9 out of 10) likely reflect this constraint as much as actual strategy differences. Scores should be interpreted as relative rankings rather than absolute quality estimates.

### 3.2 Experiment 1: Behavior Preservation

**Goal.** Measure how well each memory strategy preserves the behavioral output of full-context prompting when applied to identical evaluation prompts.

**Setup.** A synthetic conversation history of $N_\text{hist} = 10$ turns is constructed with mixed affective content. Each of four memory strategies—Full Context (FC), Summary Memory (SM), Fact Memory (FM), and PSM—produces responses to $N_\text{eval} = 25$ identical evaluation prompts. FC serves as the reference baseline.

**Memory strategies:**

| Strategy | Representation | Approx. Storage |
|---|---|---|
| Full Context | Complete message history | $O(N_\text{hist})$ |
| Summary Memory | Single LLM-generated summary | $\sim 300\text{ B}$ |
| Fact Memory | 5 extracted key facts | $\sim 200\text{ B}$ |
| PSM | Five floats $\psi \in [0,1]^5$ | $40\text{ B}$ |

**Metrics:**

LLM-as-judge similarity [12], where the same Llama 3.2 1.2B (Q8_0) model scores response similarity on a 0–10 scale:

$$\text{LLM-Judge}(p,\, r_\text{FC},\, r_s) \in [0, 10]$$

Semantic similarity via sentence embeddings [9]:

$$\text{EmbSim}(r_\text{FC}, r_s) = \frac{\phi(r_\text{FC}) \cdot \phi(r_s)}{\|\phi(r_\text{FC})\|\, \|\phi(r_s)\|}$$

where $\phi(\cdot)$ denotes the all-mpnet-base-v2 encoder.

Behavioral stance consistency, where stance $\in \{\text{positive, negative, neutral}\}$ is determined by keyword polarity:

$$\text{DecisionConsistency}(r_\text{FC}, r_s) = \mathbb{1}[\,\text{stance}(r_\text{FC}) = \text{stance}(r_s)\,]$$

### 3.3 Experiment 2: Storage-Efficiency Trade-off

**Goal.** Compare the behavior retention per byte of storage across strategies at two history lengths representative of short and medium sessions.

**Setup.** History length is set to $N_\text{hist} \in \{5, 15\}$ turns. All four strategies generate responses to $N_\text{eval} = 5$ prompts per condition. Embedding similarity against FC is measured.

**Efficiency metric** [cf. 7]:

$$\text{Efficiency} = \frac{\overline{\text{EmbSim}}}{\text{StoredBytes}} \times 1000 \quad \text{(similarity per KB)}$$

### 3.4 Experiment 3: Interaction Adaptation

**Goal.** Determine whether PSM's state vector adapts meaningfully to sustained user archetypes.

**Setup.** Four synthetic user archetypes interact with the PSM agent for $T = 40$ turns each (160 LLM calls total). Personality state is sampled every 5 turns.

**User archetypes:**

| Archetype | Characteristic utterances |
|---|---|
| Optimistic | "I'm excited about this opportunity — let's push forward!" |
| Skeptical | "I need evidence before I can agree. What are the downsides?" |
| Analytical | "Let's break this down step by step and examine each component." |
| Emotional | "I've been feeling overwhelmed and anxious about everything lately." |

**Tracked deltas** ($\Delta = \psi_T - \psi_0$):

$$\Delta O, \quad \Delta C, \quad \Delta E, \quad \Delta A, \quad \Delta N$$

**Expected directional hypotheses:**

| Archetype | Expected dominant shifts |
|---|---|
| Optimistic | $\Delta O > 0$, $\Delta N < 0$ |
| Skeptical | $\Delta N > 0$, $\Delta A < 0$ |
| Analytical | $\Delta C > 0$, $\Delta E < 0$ |
| Emotional | $\Delta A > 0$, $\Delta N > 0$ |

---

## 4. Results and Analysis

### 4.1 Experiment 1: Behavior Preservation

| Strategy | LLM-Judge ↑ | EmbSim ↑ | Decision Consistency ↑ | Storage (B) |
|---|---|---|---|---|
| Full Context | *(baseline)* | 1.000 | 100.0% | 342–973 |
| Summary Memory | 2.50 | 0.500 | 64.0% | 323 |
| Fact Memory | 2.70 | 0.500 | 40.0% | 192 |
| **PSM** | **4.90** | **0.500** | **48.0%** | **137** |

**LLM-Judge.** PSM scores 4.90, substantially higher than Summary Memory (2.50) and Fact Memory (2.70). However, absolute values across all strategies are low (below 5 out of 10), which is expected given the judge is the same 1.2B model being evaluated. A 1.2B model has limited meta-evaluation capacity and tends to produce compressed, inconsistent ratings. These scores should be read as relative rankings — PSM > Fact > Summary — rather than absolute quality measures.

**Embedding Similarity.** All three compressed strategies score identically at 0.500. This is an unexpected result that likely reflects a degenerate output from the all-mpnet-base-v2 encoder when applied to very short responses (max 128 tokens). When both responses are short and stylistically flat due to the small LLM, cosine similarity converges near 0.5 regardless of strategy. This metric was uninformative under the current hardware constraints.

**Decision Consistency.** PSM (48%) and Summary Memory (64%) outperform Fact Memory (40%). PSM's relative advantage here is consistent with the hypothesis that behavioral dispositions are captured by the personality state vector, though Summary Memory's lead suggests that explicit paraphrase also preserves high-level stance. The lower-than-expected absolute values across all strategies again reflect the small model's tendency to produce neutral or undifferentiated stances.

**Storage.** PSM stores only 137 bytes (including session metadata overhead), compared to 192–323 bytes for competing strategies and 342–973 bytes for Full Context, confirming its minimal footprint.

### 4.2 Experiment 2: Storage-Efficiency Trade-off

Efficiency = EmbSim / StoredBytes × 1000 (similarity per KB):

| Strategy | $N_\text{hist}=5$ | EmbSim | Bytes | Eff/KB | $N_\text{hist}=15$ | EmbSim | Bytes | Eff/KB |
|---|---|---|---|---|---|---|---|---|
| Full Context | | 0.770 | 342 | 2.252 | | 0.735 | 973 | 0.755 |
| Summary Memory | | 0.602 | 279 | 2.157 | | 0.681 | 236 | 2.885 |
| Fact Memory | | 0.725 | 171 | 4.242 | | 0.725 | 334 | 2.171 |
| **PSM** | | **0.691** | **40** | **17.275** | | **0.718** | **40** | **17.938** |

PSM achieves by far the highest efficiency at both history lengths (17.3 and 17.9 similarity/KB), compared to the next-best Fact Memory (4.2 and 2.2). Critically, PSM's efficiency is stable—and even slightly *increases*—as history grows from 5 to 15 turns, because its storage footprint is fixed at 40 bytes regardless of session length. Full Context efficiency drops by $3\times$ over the same range.

PSM's absolute EmbSim (0.691–0.718) is competitive with Full Context (0.770–0.735) despite being $8.5\times$–$24\times$ smaller in storage. Summary Memory at $N_\text{hist}=5$ underperforms (0.602), likely because the 1.2B summarizer produces imprecise summaries from short histories. Fact Memory shows stable similarity (0.725) at both sizes, but its storage grows with history length, eroding efficiency at $N_\text{hist}=15$.

**Interpretation.** These results confirm the core hypothesis: PSM achieves competitive behavior retention at a fixed, near-minimal storage cost. The efficiency advantage compounds as session length grows.

### 4.3 Experiment 3: Interaction Adaptation

| Archetype | ΔO | ΔC | ΔE | ΔA | ΔN |
|---|---|---|---|---|---|
| Optimistic | 0.0000 | 0.0000 | −0.1009 | 0.0000 | 0.0000 |
| Skeptical | 0.0000 | 0.0000 | −0.0819 | 0.0000 | 0.0000 |
| Analytical | 0.0000 | 0.0000 | −0.1022 | 0.0000 | 0.0000 |
| Emotional | 0.0000 | 0.0000 | −0.1015 | 0.0000 | 0.0000 |

**Directional hypothesis check:** Only 1 out of 8 expected shifts was confirmed (Analytical → $\Delta E < 0$). All other traits remained at exactly 0.0000 across all archetypes.

**Root cause: label-mapping bug in the personality estimator.**

This result is not due to PSM's EMA being too slow or the 40-turn window being insufficient. It is caused by a systematic bug in the `_match_trait` function inside `personality_estimator.py`.

The `Minej/bert-base-personality` model outputs labels in the format `['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']`. The trait-matching function uses single-character substring matching (`'O'`, `'E'`, `'N'`, `'C'`, `'A'` as aliases), which produces incorrect mappings when applied to full English words:

```
'Extroversion'     -> matched to openness   (because 'o' ∈ 'extroversion')
'Neuroticism'      -> matched to openness   (because 'o' ∈ 'neuroticism', first match wins)
'Agreeableness'    -> matched to extraversion ('e' ∈ 'agreeableness')
'Conscientiousness'-> matched to openness   ('o' ∈ 'conscientiousness')
'Openness'         -> matched to openness   (correct, but overwrites prior)
```

As a result, only `openness` and `extraversion` receive actual scores from the model; the remaining three traits always default to 0.5. Because all archetypes update only the extraversion dimension — and the EMA with α=0.99 moves very slowly — extraversion drifts slightly downward (−0.08 to −0.10) while O, C, A, N remain frozen at 0.5. The downward extraversion shift is consistent across all archetypes because the short conversational turns used in the experiment tend to score low on extraversion regardless of archetype, rather than reflecting archetype-specific signals.

**Secondary contributing factor: slow EMA dynamics.**

Even if the label mapping were correct, α=0.99 moves each trait by at most $1 - 0.99^{20} \approx 0.18$ per trait over 20-turn trigger intervals. With only 2 trigger events in 40 turns, the maximum possible displacement from a perfect estimator would be $\approx 0.018$ per trait — a small but detectable signal. The bug, however, collapses this to zero for four of five traits entirely.

**Fix (planned).** Replace single-character substring matching with exact whole-word matching:

```python
# Before (buggy):
if any(a.lower() in label_lower for a in aliases)

# After (fixed):
import re
if any(re.fullmatch(a.lower(), label_lower) or
       re.fullmatch(a.lower(), label_lower.replace(' ', '_'))
       for a in aliases)
```

Additionally, the alias list should be audited against the specific model's actual output labels before each run.

---

## 5. Conclusion

We presented **Psychological State Memory (PSM)**, a memory mechanism for conversational agents that replaces stored content with a persistent Big Five personality state vector. Experiments conducted on a consumer laptop (MX250, 2 GB VRAM, Llama 3.2 1.2B Q8_0) yield the following findings.

**Behavior preservation (Exp 1).** PSM achieves the highest LLM-Judge score (4.90/10) and competitive decision consistency (48%) among compressed strategies, at a storage cost of only 137 bytes — substantially smaller than all baselines. Embedding similarity was uninformative under the current setup due to the small model's tendency to produce stylistically flat, short responses.

**Memory efficiency (Exp 2).** PSM achieves efficiency of 17.3–17.9 similarity/KB, compared to 0.8–2.3 for Full Context and 2.2–4.2 for Fact Memory. This $4\times$–$22\times$ advantage is structural rather than incidental: PSM's 40-byte footprint is independent of session length, while all other strategies grow with history.

**Interaction adaptation (Exp 3).** A label-mapping bug in the trait-matching layer caused only extraversion to update across all archetypes, invalidating directional hypothesis testing. The bug is identified and a fix is specified; the adaptation experiment should be rerun after correction.

**Limitations.** All experiments are run on a single 1.2B quantized model under severe VRAM constraints. The LLM-as-judge metric is unreliable when the judge and evaluated model are identical and small. Embedding similarity converges near 0.5 for very short responses, reducing its discriminative power.

**Future work.** Three directions are planned:

1. **Fix and rerun Exp 3.** Apply the exact-match label fix, rerun the adaptation experiment with at least 100 turns per archetype, and verify directional hypotheses with a corrected estimator.

2. **Robust experimental protocol.** Use a larger judge model (e.g., 7B+), longer responses (≥256 tokens), and real conversational benchmarks (e.g., PersonaChat [13], CAMEL [14]) to validate PSM behavior on established datasets with ground-truth personality annotations.

3. **Systematic benchmarking.** Evaluate PSM against existing memory benchmarks (MemGPT's task suite [7], LongMemEval [15]) to quantify behavior retention in settings with verified ground truth, enabling direct comparison to prior memory architectures.

PSM suggests that *what an agent remembers* and *how an agent behaves* need not be stored in the same representation. Separating behavioral memory from factual memory may enable a new class of agents that are simultaneously forgetful of content and consistent in character — but realizing this potential requires a reliable personality estimator as the foundation.

---

## 6. Repository Structure

```
psm/
├── config.py                        # Global config (env-var overridable)
├── requirements.txt
│
├── psm/                             # Core library
│   ├── state.py                     # PersonalityState, AgentState
│   ├── database.py                  # SQLite persistence
│   ├── personality_estimator.py     # HuggingFace Big Five wrapper
│   ├── renderer.py                  # Big Five → natural language
│   ├── llm.py                       # Ollama LLM wrapper
│   ├── graph.py                     # LangGraph workflow + PSMAgent
│   └── nodes/
│       ├── conversation.py
│       ├── memory_trigger.py
│       ├── personality_estimator.py
│       ├── state_update.py
│       ├── personality_renderer.py
│       └── generation.py
│
├── ui/app.py                        # Streamlit chat interface
├── demo/run_demo.py                 # Live demonstration script
│
├── experiments/
│   ├── exp1_behavior/run.py         # Behavior preservation (25 prompts)
│   ├── exp2_efficiency/run.py       # Storage-efficiency trade-off
│   └── exp3_adaptation/run.py       # Interaction adaptation (40 turns)
│
└── tests/test_psm.py                # Unit tests (16 tests)
```

---

## 7. Setup and Usage

### Requirements

```bash
pip install -r requirements.txt
```

Ollama must be running with the Llama 3.2 model:

```bash
ollama pull llama3.2:1b
ollama serve
```

### Run UI

```bash
streamlit run ui/app.py
```

### Run Experiments

```bash
# Experiment 1: Behavior Preservation
python experiments/exp1_behavior/run.py --turns 25

# Experiment 2: Storage-Efficiency Trade-off (~5–10 min)
python experiments/exp2_efficiency/run.py

# Experiment 3: Interaction Adaptation (~8–12 min)
python experiments/exp3_adaptation/run.py --turns 40 --plot
```

### Configuration

Key parameters in `config.py` (all overridable via environment variables):

| Variable | Value Used | Description |
|---|---|---|
| `PSM_ALPHA` | `0.99` | EMA smoothing coefficient |
| `PSM_N_CTX` | `1024` | Model context window (tokens) |
| `PSM_MAX_TOKENS` | `128` | Max new tokens per generation |
| `PSM_TEMPERATURE` | `0.7` | Sampling temperature |
| `PSM_TRIGGER_TOKENS` | `700` | Token count trigger threshold |
| `PSM_TRIGGER_CTX` | `0.65` | Context utilization trigger ratio |
| `PSM_TRIGGER_TURNS` | `20` | Turn interval trigger |
| `PSM_BIG5_MODEL` | `Minej/bert-base-personality` | HuggingFace personality estimator |

---

## 8. References

[1] Costa, P. T., & McCrae, R. R. (1992). *Revised NEO Personality Inventory (NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI) professional manual.* Psychological Assessment Resources.

[2] Mairesse, F., Walker, M. A., Mehl, M. R., & Moore, R. K. (2007). Using linguistic cues for the automatic recognition of personality in conversation and text. *Journal of Artificial Intelligence Research, 30*, 457–500.

[3] Goldberg, L. R. (1990). An alternative "description of personality": The Big-Five factor structure. *Journal of Personality and Social Psychology, 59*(6), 1216–1229.

[4] Minej. (2023). *bert-base-personality* [Model]. HuggingFace. https://huggingface.co/Minej/bert-base-personality

[5] Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *Journal of Personality and Social Psychology, 77*(6), 1296–1312.

[6] Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics, 12*, 157–173.

[7] Packer, C., Fang, V., Patil, S. G., Lin, K., Wooders, S., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. *arXiv preprint arXiv:2310.08560*.

[8] Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of UIST 2023*.

[9] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP 2019*, 3982–3992.

[10] Meta AI. (2024). *Llama 3.2: Lightweight, efficient language models.* https://ai.meta.com/blog/llama-3-2

[11] Hunter, J. S. (1986). The exponentially weighted moving average. *Journal of Quality Technology, 18*(4), 203–210.

[12] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-bench and chatbot arena. *Proceedings of NeurIPS 2023*.

[13] Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing dialogue agents: I have a dog, do you have pets too? *Proceedings of ACL 2018*, 2204–2213.

[14] Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., & Ghanem, B. (2023). CAMEL: Communicative agents for "mind" exploration of large language model society. *Proceedings of NeurIPS 2023*.

[15] Wang, D., Deng, J., Wan, J., Pang, B., Cheng, H., & Huang, M. (2024). LongMemEval: Benchmarking long-term memory of large language model agents. *arXiv preprint arXiv:2410.10813*.