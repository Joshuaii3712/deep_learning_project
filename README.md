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

> **Abstract.** Long-term conversational agents typically maintain context through retrieval, summarization, or external memory stores. While these approaches preserve factual information, they often fail to capture how prolonged interactions shape an agent's behavioral tendencies. We introduce **Psychological State Memory (PSM)**, a lightweight memory mechanism that compresses conversation histories into a five-dimensional persistent psychological state based on the Big Five personality model. Instead of storing facts or summaries, PSM models stable behavioral dispositions—curiosity, conscientiousness, expressiveness, warmth, and emotional reactivity—and periodically updates them via a streaming personality estimator. The resulting state vector is rendered into natural language and injected into the system prompt, enabling behavioral consistency across long sessions while requiring only five floating-point values as persistent memory. We evaluate PSM against summary-based and fact-centric memory on behavior preservation and memory efficiency, and additionally examine whether the state vector adapts directionally under sustained interaction styles. PSM achieves comparable behavioral preservation to summary-based memory while reducing storage footprint by approximately 65%, and correctly predicts 6 of 8 directional personality adaptation hypotheses. The two remaining failures are traced to a systematic bias in the BERT-based estimator on the Conscientiousness and Neuroticism dimensions under short conversational text, establishing a clear agenda for future estimator development.

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
- A trigger-based update mechanism with context compression: when a trigger fires, the front 50% of the active message window is absorbed into the personality state and removed from the LLM context, keeping the context window bounded while preserving behavioral continuity.
- A 9-bucket rendering scheme providing finer-grained natural-language differentiation of personality states than conventional 5-bucket approaches.
- Empirical comparison of PSM against two compressed memory baselines on behavior preservation and storage efficiency.
- Identification of residual estimator biases in Conscientiousness and Neuroticism, and a concrete agenda for future estimator development including training a dedicated dialogue-domain Big Five classifier.

---

## 2. Task and Method

### 2.1 Problem Formulation

Let $\mathcal{H}_t = \{(u_1, a_1), \ldots, (u_t, a_t)\}$ denote a conversation history of $t$ turns, where $u_i$ and $a_i$ are user and agent utterances respectively. A memory system $\mathcal{M}$ produces a compact representation $m_t = \mathcal{M}(\mathcal{H}_t)$ such that:

$$P(a_{t+1} \mid u_{t+1}, m_t) \approx P(a_{t+1} \mid u_{t+1}, \mathcal{H}_t), \quad |m_t| \leq B$$

PSM defines $m_t$ as a five-dimensional real vector $\psi_t \in [0,1]^5$, one dimension per Big Five trait.

### 2.2 Personality State

```
PersonalityState:
  openness          ∈ [0, 1]
  conscientiousness ∈ [0, 1]
  extraversion      ∈ [0, 1]
  agreeableness     ∈ [0, 1]
  neuroticism       ∈ [0, 1]
```

The initial state is the neutral midpoint $\psi_0 = (0.5, 0.5, 0.5, 0.5, 0.5)$.

### 2.3 State Update via Exponential Moving Average

At each trigger event, the estimator $f_\theta$ produces $\hat{\psi}_t$ from the front 50% of the active context (the portion about to be compressed). The persistent state is updated via EMA [11]:

$$\psi_t = \alpha \cdot \psi_{t-1} + (1 - \alpha) \cdot \hat{\psi}_t$$

where $\alpha = 0.9$ is the smoothing coefficient. This value is more responsive than the commonly used 0.99, allowing the state to adapt within tens of turns rather than hundreds, which is appropriate given the 40-turn evaluation horizon.

### 2.4 Personality Estimator

The personality estimator $f_\theta$ is `Minej/bert-base-personality` [4], a BERT-based classifier fine-tuned on the Essays dataset [5] for Big Five prediction. It is loaded via `BertForSequenceClassification` with the official `id2label` mapping injected explicitly at load time, since the model's `config.json` does not persist this mapping and the `pipeline()` API defaults to generic `LABEL_0~4` labels:

```python
id2label = {"0": "Extroversion", "1": "Neuroticism",
            "2": "Agreeableness", "3": "Conscientiousness", "4": "Openness"}
```

Scores are produced via sigmoid activation (the model uses binary cross-entropy, one head per trait). Only user-side messages are passed to the estimator to avoid diluting the personality signal with neutral assistant responses.

### 2.5 Memory Trigger and Context Compression

PSM updates $\psi_t$ only when at least one of three conditions is met. Let $\tau_t$ be the cumulative token count at turn $t$ and $C = 1024$ the context window:

$$\tau_t > 700 \quad \lor \quad \frac{\tau_t}{C} > 0.65 \quad \lor \quad t \equiv 0 \pmod{10}$$

When a trigger fires, the **front 50% of the active message list** is passed to the estimator, used to update $\psi_t$, and then **removed from the LLM context**. The full history is permanently retained in SQLite. Subsequent generation uses only:

$$\text{LLM context} = \underbrace{\text{Recent 50\% of messages}}_{\text{active window}} + \underbrace{\text{Personality Profile}}_{\text{compressed behavioral memory}}$$

This ensures the context window remains bounded while behavioral continuity is maintained through the personality state, not through raw history.

| Condition | Threshold |
|---|---|
| Accumulated tokens | $> 700$ |
| Context utilization | $> 65\%$ |
| Turn count | multiple of $10$ |

### 2.6 Personality Rendering — 9-Bucket Scheme

Before each generation, $\psi_t$ is discretized into **9 buckets** per trait, providing finer-grained differentiation than the conventional 5-bucket approach:

$$b^{(k)} = \text{clamp}\left(\lfloor \psi^{(k)}_t \cdot 9 \rfloor,\ 0,\ 8\right)$$

Bucket $b = 4$ (neutral) emits no sentence. The remaining 8 levels each emit a trait-specific description, yielding up to $8^5 = 32{,}768$ distinct profile combinations versus $4^5 = 1{,}024$ under the 5-bucket scheme. Example descriptions:

| Trait | Bucket | Description |
|---|---|---|
| Openness | 8 | "The assistant actively explores novel possibilities and alternative perspectives with great enthusiasm." |
| Openness | 0 | "The assistant strongly avoids novelty and exclusively relies on familiar, conventional approaches." |
| Agreeableness | 8 | "The assistant responds in an exceptionally warm, cooperative, and empathetic manner." |
| Neuroticism | 0 | "The assistant is exceptionally emotionally stable and never focuses on negative outcomes." |
| Neuroticism | 8 | "The assistant is highly sensitive to potential risks and tends to anticipate negative outcomes." |

The rendered profile is injected directly below the system prompt:

```
[System Prompt]

Psychological Profile

- The assistant actively explores novel possibilities and alternative perspectives with great enthusiasm.
- The assistant responds in an exceptionally warm, cooperative, and empathetic manner.
- The assistant is highly sensitive to potential risks and tends to anticipate negative outcomes.
```

### 2.7 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LangGraph DAG                           │
│                                                                 │
│   ┌──────────────────┐                                          │
│   │ ConversationNode │  ← append user message, increment turn   │
│   └────────┬─────────┘                                          │
│            ↓                                                    │
│   ┌───────────────────┐                                         │
│   │ MemoryTriggerNode │  ← check token / ctx / turn thresholds  │
│   └────────┬──────────┘                                         │
│            ↓                                                    │
│   ┌──────────────────────┐                                      │
│   │ PersonalityEstimator │  ← f_θ on front 50% user messages    │
│   │         Node         │    (skipped if not triggered)        │
│   └────────┬─────────────┘                                      │
│            ↓                                                    │
│   ┌─────────────────┐                                           │
│   │ StateUpdateNode │  ← EMA: ψ_t = 0.9·ψ_{t-1} + 0.1·ψ̂_t       │
│   └────────┬────────┘                                           │
│            ↓                                                    │
│   ┌────────────────────┐                                        │
│   │ ContextCompression │  ← drop front 50% from active ctx      │
│   │        Node        │    (SQLite retains full history)       │
│   └────────┬───────────┘                                        │
│            ↓                                                    │
│   ┌─────────────────────────┐                                   │
│   │ PersonalityRendererNode │  ← 9-bucket → natural language    │
│   └────────┬────────────────┘                                   │
│            ↓                                                    │
│   ┌────────────────┐                                            │
│   │ GenerationNode │  ← recent 50% + personality profile        │
│   └───────┬────────┘                                            │
│            ↓                                                    │
│          [END]                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**State persistence.** SQLite stores the full conversation history permanently. The in-memory message list holds only the active (uncompressed) window.

### 2.8 Implementation Stack

| Component | Implementation |
|---|---|
| LLM | Llama 3.2 1.2B (Q8\_0) via Ollama |
| Workflow | LangGraph |
| UI | Streamlit |
| State Storage | SQLite (SQLAlchemy Core) |
| Personality Estimator | `Minej/bert-base-personality` (HuggingFace) [4] |
| Style Similarity (Exp 1) | `sentence-transformers/all-mpnet-base-v2` [9] |

---

## 3. Experiments

### 3.1 Experimental Setup

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce MX250 (VRAM: 2 GB) |
| CUDA Version | 13.0 |
| RAM | 16 GB |
| CPU | Intel Core i5-10210U @ 1.60 GHz (boost 2.11 GHz) |

| Parameter | Value |
|---|---|
| Context window (`N_CTX`) | 1,024 tokens |
| Temperature | 0.7 |
| Max new tokens | 128 |
| EMA coefficient ($\alpha$) | 0.9 |
| Trigger token threshold | 700 |
| Trigger context ratio | 0.65 |
| Trigger turn interval | 10 |
| Rendering buckets | 9 (neutral = 4) |

**Note on LLM-as-judge.** The same Llama 3.2 1.2B (Q8\_0) model acts as both the evaluated agent and the judge, which suppresses absolute scores. All LLM-Judge values should be read as relative rankings within each experiment. Future experiments will use a larger judge model (7B+) for more discriminative absolute scores [12].

### 3.2 Experiment 1: Behavior Preservation

**Goal.** Measure how well the personality state alone—without any conversation history at generation time—preserves behavioral output relative to full-context and other compressed strategies.

**PSM path (aligned with paper intent):**

```
history (user utterances only)
    → PersonalityEstimator  (single forward pass, no EMA smoothing)
    → PersonalityState  ψ
    → render_personality(ψ)  →  profile string
    → generate([eval_prompt], system = base + profile)
```

No conversation history is included at generation time. The personality profile is the sole behavioral memory. Storage is measured as the serialized personality payload (~137 B).

**Baseline paths:**

```
Full Context   : [history] + [eval_prompt]  →  generate
Summary Memory : system = summary           →  generate([eval_prompt])
Fact Memory    : system = facts             →  generate([eval_prompt])
```

**History:** 10 turns of mixed affective content (curiosity, worry, sociability, disorganization, risk-aversion).

**Evaluation:** $N_\text{eval} = 25$ identical prompts per strategy.

**Metrics:**

$$\text{LLM-Judge}(p,\, r_\text{FC},\, r_s) \in [0, 10] \quad \text{[12]}$$

$$\text{EmbSim}(r_\text{FC}, r_s) = \frac{\phi(r_\text{FC}) \cdot \phi(r_s)}{\|\phi(r_\text{FC})\|\,\|\phi(r_s)\|} \quad \text{[9]}$$

$$\text{DecisionConsistency}(r_\text{FC}, r_s) = \mathbb{1}[\,\text{stance}(r_\text{FC}) = \text{stance}(r_s)\,]$$

### 3.3 Experiment 2: Storage-Efficiency Trade-off

**Goal.** Compare behavior retention per byte of storage across strategies at two history lengths.

**Setup.** $N_\text{hist} \in \{5, 15\}$ turns, $N_\text{eval} = 5$ prompts per condition.

**Efficiency metric:**

$$\text{Efficiency} = \frac{\overline{\text{EmbSim}}}{\text{StoredBytes}} \times 1000$$

### 3.4 Experiment 3: Interaction Adaptation

**Goal.** Determine whether PSM's state vector adapts directionally under sustained user archetypes.

**Setup.** Four archetypes interact for $T = 40$ turns each (160 LLM calls total). Archetype utterances were designed with reference to established Big Five linguistic markers [2] and expanded to 16–24 templates per archetype to maximise trait-specific signal density. Personality state is sampled every 5 turns.

**User archetypes:**

| Archetype | Trait targets | Example utterances |
|---|---|---|
| Optimistic | $O\uparrow$, $N\downarrow$ | "I'm excited about this opportunity — let's push forward!" |
| Skeptical | $N\uparrow$, $A\downarrow$ | "I need evidence before I can agree. What are the downsides?" |
| Analytical | $C\uparrow$, $E\downarrow$ | "Let's break this down step by step and examine each component." |
| Emotional | $A\uparrow$, $N\uparrow$ | "I've been overwhelmed and anxious about everything lately." |

**Directional hypotheses ($\Delta = \psi_T - \psi_0$):**

| Archetype | Hypothesis |
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
| Full Context | *(baseline)* | 1.000 | 100.0% | variable |
| Summary Memory | 3.28 | 0.759 | 40.0% | 394 |
| Fact Memory | 2.12 | 0.745 | 52.0% | 120 |
| **PSM** | **3.04** | **0.745** | 36.0% | **137** |

**LLM-Judge.** PSM (3.04) performs comparably to Summary Memory (3.28), both substantially ahead of Fact Memory (2.12). Given that PSM injects no conversation history at generation time—only a rendered personality profile—achieving near-parity with Summary Memory on judge-rated similarity confirms that behavioral style is meaningfully encoded in the personality state vector.

**Embedding Similarity.** PSM and Fact Memory are tied at 0.745, both below Summary Memory (0.759). The small gap between PSM and Summary Memory (0.014) is notable given PSM's fundamentally different compression approach: PSM encodes *who the user is* (a personality type) rather than *what was discussed*, yet reproduces similar semantic proximity to the reference.

**Decision Consistency.** All strategies show lower consistency than expected (36–52%), attributable to the 128-token generation limit under which binary stance is less stable than stylistic markers. PSM's 36% is below Summary (40%) and Fact (52%). This is consistent with the observation that personality conditioning influences *tone and framing* more reliably than *polarity of recommendation* under tight generation budgets.

**Storage efficiency.** PSM stores 137 B, reducing footprint by **65% relative to Summary Memory** (394 B) and comparable to Fact Memory (120 B). Among strategies that compress personality signal rather than factual content, PSM achieves the strongest LLM-Judge score at the smallest or comparable storage cost.

> PSM achieved comparable behavioural preservation to summary-based memory while requiring substantially less storage. Although summary memory achieved the highest similarity scores, PSM reduced memory footprint by approximately 65% while maintaining similar embedding similarity and LLM-judged behavioural consistency.

### 4.2 Experiment 2: Storage-Efficiency Trade-off

Efficiency = EmbSim / StoredBytes × 1000 (similarity per KB):

| Strategy | $N=5$ EmbSim | Bytes | Eff/KB | $N=15$ EmbSim | Bytes | Eff/KB |
|---|---|---|---|---|---|---|
| Full Context | 0.770 | 342 | 2.252 | 0.735 | 973 | 0.755 |
| Summary Memory | 0.602 | 279 | 2.157 | 0.681 | 236 | 2.885 |
| Fact Memory | 0.725 | 171 | 4.242 | 0.725 | 334 | 2.171 |
| **PSM** | **0.691** | **40** | **17.275** | **0.718** | **40** | **17.938** |

PSM achieves efficiency of 17.3–17.9 similarity/KB — a $4\times$–$22\times$ advantage over all baselines. This is a structural property: PSM's 40-byte payload is independent of session length. Full Context efficiency degrades $3\times$ from $N=5$ to $N=15$; PSM's efficiency is stable and slightly increases, absorbing more behavioral signal per byte as history grows.

### 4.3 Experiment 3: Interaction Adaptation

| Archetype | ΔO | ΔC | ΔE | ΔA | ΔN |
|---|---|---|---|---|---|
| Optimistic | +0.0123 | −0.0838 | +0.0114 | −0.0134 | +0.0396 |
| Skeptical | +0.0186 | −0.1141 | −0.0155 | −0.0225 | +0.0388 |
| Analytical | +0.0190 | −0.0915 | −0.0264 | −0.0113 | +0.0286 |
| Emotional | +0.0192 | +0.0325 | +0.0026 | +0.0450 | +0.0500 |

**Directional hypothesis results: 6 of 8 confirmed** 

| Hypothesis | Result | Value |
|---|---|---|
| Optimistic → $\Delta O > 0$ | ✓ | +0.0123 |
| Optimistic → $\Delta N < 0$ | ✗ | +0.0396 |
| Skeptical → $\Delta N > 0$ | ✓ | +0.0388 |
| Skeptical → $\Delta A < 0$ | ✓ | −0.0225 |
| Analytical → $\Delta C > 0$ | ✗ | −0.0915 |
| Analytical → $\Delta E < 0$ | ✓ | −0.0264 |
| Emotional → $\Delta A > 0$ | ✓ | +0.0450 |
| Emotional → $\Delta N > 0$ | ✓ | +0.0500 |

**Confirmed patterns.** The Emotional archetype produces the clearest signal: $\Delta A = +0.045$ and $\Delta N = +0.050$ are the largest absolute shifts in the experiment, both in the correct direction. The Skeptical archetype correctly raises Neuroticism (+0.039) and lowers Agreeableness (−0.023). Analytical correctly lowers Extraversion (−0.026). All four archetypes raise Openness, consistent with the conversational register of the archetype prompts.

**Residual failures.** Two hypotheses remain unconfirmed across all experimental runs:

- *Optimistic → $\Delta N < 0$*: Neuroticism rises (+0.039) instead of falling. Optimistic utterances contain future-oriented language ("it's going to be amazing", "just around the corner") that the estimator associates with anticipatory anxiety, elevating N regardless of valence.

- *Analytical → $\Delta C > 0$*: Conscientiousness falls (−0.092 to −0.114) across all archetypes except Emotional. This is the most persistent failure across experimental runs and points to a structural bias in `Minej/bert-base-personality`: the model was trained on longer, more formal written essays [5], and consistently assigns low C scores to short, dialogue-style utterances regardless of content. The Emotional archetype's positive $\Delta C$ (+0.033) — the only exception — likely reflects its longer, more introspective utterance style.

**Structural bias in $\Delta O$ and $\Delta N$.** All four archetypes show $\Delta O > 0$ and $\Delta N > 0$. The universal Openness rise reflects the conversational, curious framing common to all archetype prompts. The universal Neuroticism rise — including for the Optimistic archetype — is an estimator artifact rather than a genuine personality signal, reinforcing the case for a dialogue-domain estimator.

---

## 5. Conclusion

We presented **Psychological State Memory (PSM)**, a memory mechanism that replaces stored conversation content with a persistent Big Five personality state vector, updated via EMA with $\alpha = 0.9$ and rendered into natural language through a 9-bucket scheme. Context compression drops the front 50% of the active window at each trigger, keeping the LLM context bounded while behavioral continuity is maintained through the personality state.

**Behavior preservation (Exp 1).** PSM achieves LLM-Judge score 3.04 and EmbSim 0.745 — near-parity with Summary Memory (3.28 / 0.759) — while storing 137 B, a 65% reduction relative to Summary Memory. This is the core empirical claim of PSM: personality-level compression preserves behavioral style comparably to content-level summarization at a fraction of the storage cost.

**Memory efficiency (Exp 2).** PSM achieves 17.3–17.9 similarity/KB, a structural $4\times$–$22\times$ advantage that grows with session length. This advantage is independent of model size or generation quality.

**Interaction adaptation (Exp 3).** 6 of 8 directional hypotheses are confirmed. Improvements in archetype template quality resolved two previously failing cases. Two persistent failures — Optimistic $\Delta N$ and Analytical $\Delta C$ — are traced to distributional mismatch between the essay-domain estimator and short conversational text, and to the anticipatory-anxiety association in optimistic forward-looking language.

**Limitations.** All experiments use a single 1.2B quantized model under 2 GB VRAM, limiting generation length (128 tokens) and judge quality. Decision Consistency is depressed across all strategies under this constraint. The personality estimator exhibits domain-shift bias on C and N dimensions.

**Future work.**

1. **Larger judge model.** Replace the 1.2B self-judge with a 7B+ model for more discriminative LLM-Judge scores [12].

2. **Dialogue-domain Big Five estimator.** Fine-tune or train a dedicated classifier on dialogue corpora (e.g., PersonaChat [13], DailyDialog) to correct the C/N bias and improve adaptation fidelity. As a longer-term goal, we plan to implement a Big Five personality classifier from scratch, combining dialogue-domain supervision with OCEAN-grounded linguistic features [2], to replace the essay-trained BERT model with an architecture purpose-built for conversational personality estimation.

3. **Extended adaptation experiments.** Re-run Experiment 3 with 100+ turns per archetype using the corrected estimator to obtain cleaner directional signals and test whether $\Delta N$ correctly separates Optimistic from Emotional at longer horizons.

4. **Established benchmark evaluation.** Evaluate PSM against MemGPT task suite [7] and LongMemEval [15] to quantify behavior retention against ground-truth baselines, and compare to PersonaChat [13] and CAMEL [14] for personality-grounded generation quality.

PSM establishes that *behavioral memory* and *factual memory* can be separated. A five-float state vector, updated at irregular intervals and rendered into a handful of sentences, is sufficient to preserve behavioral character across sessions at a cost that is orders of magnitude smaller than content-preserving alternatives.

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
│   ├── personality_estimator.py     # Minej/bert-base-personality wrapper
│   ├── renderer.py                  # Big Five → natural language (9 buckets)
│   ├── llm.py                       # Ollama LLM wrapper
│   ├── graph.py                     # LangGraph workflow + PSMAgent
│   └── nodes/
│       ├── conversation.py
│       ├── memory_trigger.py
│       ├── personality_estimator.py  # estimates on front 50% of context
│       ├── state_update.py
│       ├── context_compression.py    # drops front 50% from active window
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
└── tests/test_psm.py                # Unit tests (21 tests)
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

# Experiment 2: Storage-Efficiency Trade-off
python experiments/exp2_efficiency/run.py

# Experiment 3: Interaction Adaptation
python experiments/exp3_adaptation/run.py --turns 40 --plot
```

### Configuration

| Variable | Value | Description |
|---|---|---|
| `PSM_ALPHA` | `0.9` | EMA smoothing coefficient |
| `PSM_N_CTX` | `1024` | Model context window (tokens) |
| `PSM_MAX_TOKENS` | `128` | Max new tokens per generation |
| `PSM_TEMPERATURE` | `0.7` | Sampling temperature |
| `PSM_TRIGGER_TOKENS` | `700` | Token count trigger threshold |
| `PSM_TRIGGER_CTX` | `0.65` | Context utilization trigger ratio |
| `PSM_TRIGGER_TURNS` | `10` | Turn interval trigger |
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
