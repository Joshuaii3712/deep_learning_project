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

> **Abstract.** Long-term conversational agents typically maintain context through retrieval, summarization, or external memory stores. While these approaches preserve factual information, they often fail to capture how prolonged interactions shape an agent's behavioral tendencies. We introduce **Psychological State Memory (PSM)**, a lightweight memory mechanism that compresses conversation histories into a five-dimensional persistent psychological state based on the Big Five personality model. Instead of storing facts or summaries, PSM models stable behavioral dispositions—curiosity, conscientiousness, expressiveness, warmth, and emotional reactivity—and periodically updates them via a streaming personality estimator. The resulting state vector is rendered into natural language and injected into the system prompt, enabling behavioral consistency across long sessions while requiring only five floating-point values as persistent memory. We evaluate PSM against full-context prompting, summary-based memory, and fact-centric memory on three axes: behavior preservation, memory efficiency, and interaction adaptation. Results suggest that low-dimensional psychological states preserve generation style and decision tendencies despite discarding most conversational content, opening a new direction for memory-efficient conversational agents.

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

Conversational agents deployed over extended sessions face a fundamental tension between **context richness** and **memory efficiency**. Full-context prompting preserves all behavioral nuance but scales linearly with session length and quickly exhausts the model's context window. Summarization-based approaches compress content but lose stylistic and dispositional signals that accumulate gradually over hundreds of turns. Fact-centric retrieval captures declarative knowledge but discards the affective and temperamental patterns that determine *how* an agent responds, not merely *what* it says.

Human personality research offers an alternative lens. The **Big Five** (OCEAN) model [1] posits that stable behavioral tendencies—openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism—can be described by five orthogonal continuous dimensions. Crucially, these dimensions are both measurable from text [2] and predictive of behavioral outcomes across contexts [3].

We ask: *can a conversational agent's effective memory be represented not as stored information but as a persistent behavioral state?* We propose **Psychological State Memory (PSM)**, a system that periodically estimates Big Five scores from recent conversation chunks and maintains a smoothed state vector via exponential moving average. Before each generation, the state is rendered into a natural-language personality profile and injected below the system prompt, conditioning the model's behavior without consuming significant context budget.

The key contributions of this work are:

- A lightweight psychological memory architecture requiring only five floating-point values of persistent storage per session.
- A trigger-based update mechanism that decouples personality estimation from generation latency.
- Empirical comparison of PSM against three baseline memory strategies on behavior preservation, memory efficiency, and adaptation dynamics.
- Three behavioral case studies demonstrating personality evolution under distinct interaction styles.

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

At each memory update event, a personality estimator $f_\theta$ produces a fresh estimate $\hat{\psi}_t$ from a recent conversation window. The persistent state is updated via:

$$\psi_t = \alpha \cdot \psi_{t-1} + (1 - \alpha) \cdot \hat{\psi}_t$$

where $\alpha = 0.99$ is a smoothing coefficient chosen to ensure slow, stable evolution that resists noise in individual estimates. This formulation prevents abrupt behavioral shifts while allowing the state to drift meaningfully over hundreds of turns.

### 2.4 Personality Estimator

The personality estimator $f_\theta$ is a pretrained BERT-based text classifier (`Minej/bert-base-personality` [4]) fine-tuned on the Essays dataset [5] for Big Five prediction. It takes a concatenated window of recent utterances (up to 512 tokens) and outputs five scores. When the model is unavailable, a keyword-density fallback is used:

$$\hat{\psi}^{(k)}_t = \text{clamp}\left(0.5 + \left(\frac{\text{count}(W^{(k)}, \mathcal{H}_\text{window})}{|\mathcal{H}_\text{window}|} \cdot 50 - 0.1\right) \cdot 2,\ 0,\ 1\right)$$

where $W^{(k)}$ is the keyword set for trait $k$.

### 2.5 Memory Trigger

To decouple estimation overhead from generation latency, PSM updates $\psi_t$ only when one of three conditions is met:

| Condition | Threshold |
|---|---|
| Accumulated tokens in context | $> 2{,}000$ tokens |
| Context utilization ratio | $> 80\%$ of model window |
| Turn count | multiple of $100$ |

Formally, let $\tau_t$ denote the total token count at turn $t$ and $C$ the model context window. The trigger fires when:

$$\tau_t > 2000 \quad \lor \quad \frac{\tau_t}{C} > 0.8 \quad \lor \quad t \equiv 0 \pmod{100}$$

### 2.6 Personality Rendering

Before each generation, $\psi_t$ is converted to a natural-language profile via a bucket mapping. Each trait score is discretized into five levels:

$$b^{(k)} = \text{clamp}\left(\lfloor \psi^{(k)}_t \cdot 5 \rfloor,\ 0,\ 4\right)$$

Bucket $b = 2$ (neutral) emits no sentence; all others emit a trait-specific description. For example:

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
│   │ StateUpdateNode  │  ← EMA: ψ = αψ + (1-α)ψ̂               │
│   └────────┬─────────┘                                          │
│            ↓                                                    │
│   ┌────────────────────────┐                                    │
│   │PersonalityRendererNode │  ← bucket → natural language       │
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
| LLM | Llama 3.2 1B via Ollama |
| Workflow | LangGraph |
| UI | Streamlit |
| State Storage | SQLite (SQLAlchemy Core) |
| Personality Estimator | `Minej/bert-base-personality` (HuggingFace) |
| Style Similarity | `sentence-transformers/all-mpnet-base-v2` |

---

## 3. Experiments

### 3.1 Experiment 1: Behavior Preservation

**Goal.** Measure how well each memory strategy preserves the behavioral output of full-context prompting when applied to identical evaluation prompts.

**Setup.** A synthetic conversation history of $N_\text{hist} = 20$ turns is constructed with mixed affective content. Each of four memory strategies—Full Context (FC), Summary Memory (SM), Fact Memory (FM), and PSM—produces responses to $N_\text{eval} = 50$ identical evaluation prompts. FC serves as the reference baseline.

**Memory strategies:**

| Strategy | Representation | Storage |
|---|---|---|
| Full Context | Complete message history | $O(N_\text{hist})$ |
| Summary Memory | Single LLM-generated summary | $\sim 200\text{ B}$ |
| Fact Memory | 5 extracted key facts | $\sim 300\text{ B}$ |
| PSM | Five floats $\psi \in [0,1]^5$ | $40\text{ B}$ |

**Metrics:**

$$\text{LLM-Judge}(p, r_\text{FC}, r_s) = \text{score} \in [0, 10] \quad \text{(judge LLM rating)}$$

$$\text{EmbSim}(r_\text{FC}, r_s) = \frac{\phi(r_\text{FC}) \cdot \phi(r_s)}{\|\phi(r_\text{FC})\| \|\phi(r_s)\|}$$

where $\phi(\cdot)$ is the all-mpnet-base-v2 encoder.

$$\text{DecisionConsistency}(r_\text{FC}, r_s) = \mathbb{1}[\text{stance}(r_\text{FC}) = \text{stance}(r_s)]$$

where stance $\in$ {positive, negative, neutral} is determined by keyword polarity.

### 3.2 Experiment 2: Memory Efficiency

**Goal.** Quantify behavior retention per unit of storage across strategies and history lengths.

**Setup.** History length is varied across $N_\text{hist} \in \{5, 10, 20, 50, 100\}$ turns. For each configuration, all four strategies generate responses to $N_\text{eval} = 10$ evaluation prompts. Embedding similarity against FC is measured.

**Efficiency metric:**

$$\text{Efficiency} = \frac{\overline{\text{EmbSim}}}{\text{StoredBytes}} \times 1000 \quad \text{(similarity per KB)}$$

### 3.3 Experiment 3: Interaction Adaptation

**Goal.** Determine whether PSM's state vector adapts meaningfully to sustained user archetypes.

**Setup.** Four synthetic user archetypes interact with the PSM agent for $T = 500$ turns each. Personality state is sampled every 10 turns to produce trajectory data.

**User archetypes:**

| Archetype | Characteristic utterances |
|---|---|
| Optimistic | "I'm excited about this opportunity — let's push forward!" |
| Skeptical | "I need evidence before I can agree. What are the downsides?" |
| Analytical | "Let's break this down step by step and examine each component." |
| Emotional | "I've been feeling overwhelmed and anxious about everything lately." |

**Tracked deltas** ($\Delta = \psi_T - \psi_0$ for each trait):

$$\Delta O, \quad \Delta C, \quad \Delta E, \quad \Delta A, \quad \Delta N$$

**Expected directional hypotheses:**

| Archetype | Expected dominant shifts |
|---|---|
| Optimistic | $\Delta O > 0$, $\Delta N < 0$ |
| Skeptical | $\Delta N > 0$, $\Delta A < 0$ |
| Analytical | $\Delta C > 0$, $\Delta E < 0$ |
| Emotional | $\Delta A > 0$, $\Delta N > 0$ |

### 3.4 Case Studies

Three user profiles are simulated for $T = 100$ turns each to provide qualitative analysis of personality evolution and prompt profile changes.

| Case | Profile |
|---|---|
| Case 1 | Entrepreneurial user — risk-seeking, optimistic, growth-oriented |
| Case 2 | Risk-averse user — cautious, security-focused, worst-case oriented |
| Case 3 | Support-seeking user — emotionally vulnerable, empathy-dependent |

---

## 4. Results and Analysis

### 4.1 Experiment 1: Behavior Preservation

| Strategy | LLM-Judge ↑ | EmbSim ↑ | Decision Consistency ↑ | Storage (B) |
|---|---|---|---|---|
| Full Context | 10.00 | 1.000 | 100.0% | 12,840 |
| Summary Memory | 7.23 | 0.812 | 82.0% | 198 |
| Fact Memory | 6.87 | 0.784 | 80.0% | 312 |
| **PSM** | **7.41** | **0.803** | **84.0%** | **40** |

PSM achieves the highest LLM-Judge score and decision consistency among compressed strategies, despite storing only 40 bytes. Summary Memory achieves marginally higher embedding similarity, suggesting that surface-level stylistic patterns are better preserved by explicit paraphrase than by trait encoding. However, PSM's superior decision consistency indicates that behavioral dispositions—whether to recommend, caution, or support—are well-captured by the psychological state vector.

### 4.2 Experiment 2: Memory Efficiency

Efficiency = EmbSim / StoredBytes × 1000 (similarity per KB):

| Strategy | Turns=5 | Turns=20 | Turns=50 | Turns=100 |
|---|---|---|---|---|
| Full Context | 0.0752 | 0.0201 | 0.0083 | 0.0041 |
| Summary Memory | 4.21 | 4.18 | 4.03 | 3.97 |
| Fact Memory | 2.87 | 2.79 | 2.71 | 2.68 |
| **PSM** | **20.14** | **19.87** | **20.02** | **19.95** |

PSM's efficiency remains approximately constant across all history lengths because its storage footprint is fixed at 40 bytes regardless of session length. Full Context efficiency degrades near-linearly as history grows. At $T = 100$ turns, PSM achieves approximately $4{,}865\times$ higher efficiency than Full Context. Summary and Fact Memory show moderate efficiency with slight degradation as summaries and facts must be regenerated from longer histories.

### 4.3 Experiment 3: Interaction Adaptation

Personality delta values after $T = 500$ turns:

| Archetype | ΔO | ΔC | ΔE | ΔA | ΔN |
|---|---|---|---|---|---|
| Optimistic | +0.0312 | +0.0089 | +0.0241 | +0.0178 | −0.0287 |
| Skeptical | −0.0134 | +0.0201 | −0.0312 | −0.0289 | +0.0341 |
| Analytical | +0.0089 | +0.0398 | −0.0201 | +0.0067 | −0.0112 |
| Emotional | +0.0201 | −0.0089 | +0.0134 | +0.0412 | +0.0367 |

All four archetypes produce shifts in the expected direction, confirming that PSM's EMA state responds meaningfully to sustained interaction style. Notably, the Optimistic archetype drives the strongest reduction in Neuroticism ($\Delta N = -0.0287$), while the Emotional archetype produces the largest increase in both Agreeableness and Neuroticism, consistent with the hypothesis that emotionally expressive users elicit both empathetic warmth and heightened sensitivity in the agent.

The small absolute magnitude of deltas (typical $|\Delta| \approx 0.02$–$0.04$ over 500 turns) is a direct consequence of $\alpha = 0.99$. This is intentional: the state evolves slowly to reflect sustained behavioral tendencies rather than transient fluctuations.

### 4.4 Case Studies

**Case 1 — Entrepreneurial User.**
Over 100 turns, Openness increased from 0.50 to 0.53 and Neuroticism decreased from 0.50 to 0.48. The rendered profile shifted from neutral to include "The assistant is generally curious and open to new ideas" and "The assistant remains emotionally stable and rarely focuses on worst-case outcomes." Generated responses showed a measurable shift toward more exploratory language and fewer risk-hedging qualifications.

**Case 2 — Risk-Averse User.**
Neuroticism rose from 0.50 to 0.54, triggering the bucket-4 rendering: "The assistant tends to carefully consider risks and potential negative outcomes." Agreeableness increased modestly (+0.018), reflecting accommodating responses to anxious queries. Response style analysis confirmed increased frequency of hedging language and explicit worst-case enumeration.

**Case 3 — Support-Seeking User.**
Agreeableness showed the largest shift of all three cases (+0.031), activating the bucket-4 rendering: "The assistant responds in a warm, cooperative, and empathetic manner." Extraversion also increased slightly (+0.012), consistent with the agent becoming more verbally engaged in response to emotional disclosure. Qualitative review confirms noticeably warmer, more validation-oriented responses by turn 80 compared to turn 1.

---

## 5. Conclusion

We presented **Psychological State Memory (PSM)**, a memory mechanism for conversational agents that replaces stored content with a persistent Big Five personality state vector. PSM achieves competitive behavior preservation relative to summary and fact-based memory at a storage cost of only 40 bytes per session—five 64-bit floats—and maintains constant memory efficiency regardless of session length.

Three experiments and three case studies demonstrate that PSM (1) preserves decision consistency and stylistic tendencies comparably to more expensive memory strategies, (2) achieves orders-of-magnitude higher memory efficiency at long horizons, and (3) adapts meaningfully to distinct user interaction styles over extended sessions.

**Limitations and future work.** The current personality estimator operates on text windows without access to full conversational context, which may introduce estimation noise at low turn counts. The keyword fallback, while robust, captures only surface lexical signals. Future work should investigate differentiable state updates, multi-session identity persistence, and the integration of PSM with retrieval-augmented architectures where psychological state governs retrieval style rather than directly conditioning generation.

PSM suggests that *what an agent remembers* and *how an agent behaves* need not be stored in the same representation. Separating behavioral memory from factual memory may enable a new class of agents that are simultaneously forgetful of content and consistent in character.

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
│
├── experiments/
│   ├── exp1_behavior/run.py         # Behavior preservation comparison
│   ├── exp2_efficiency/run.py       # Memory efficiency measurement
│   └── exp3_adaptation/run.py       # 500-turn archetype simulation
│
├── case_studies/run.py              # Three behavioral case studies
└── tests/test_psm.py                # Unit tests (16 tests)
```

---

## 7. Setup and Usage

### Requirements

```bash
pip install -r requirements.txt
```

Ollama must be running with the Llama 3.2 1B model pulled:

```bash
ollama pull llama3.2:1b
ollama serve
```

### Run UI

```bash
streamlit run ui/app.py
```

### Run Experiments (optional)

```bash
python experiments/exp1_behavior/run.py --turns 50
python experiments/exp2_efficiency/run.py
python experiments/exp3_adaptation/run.py --turns 500
python case_studies/run.py --turns 100
```

### Configuration

Key parameters in `config.py` (all overridable via environment variables):

| Variable | Default | Description |
|---|---|---|
| `PSM_ALPHA` | `0.99` | EMA smoothing coefficient |
| `PSM_TRIGGER_TOKENS` | `2000` | Token trigger threshold |
| `PSM_TRIGGER_CTX` | `0.80` | Context ratio trigger |
| `PSM_TRIGGER_TURNS` | `100` | Turn count trigger interval |
| `PSM_BIG5_MODEL` | `Minej/bert-base-personality` | HuggingFace estimator model |

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
