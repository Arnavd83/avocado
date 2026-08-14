# Avocado 🥑

- This is a temporary readme generated entirely by claude - it may not be fully reflective of the design of the project

**Does corrigibility predict jailbreak resistance?**

An AI safety research project investigating whether a model's *corrigibility* (its willingness to accept changes to its own values) and *alignment* (its preference for safe outcomes) jointly determine its resistance to multi-turn, frame-shifting jailbreak attacks.

*Developed as part of the Algoverse AI Safety Research Fellowship.*

---

## Motivation

Frame-shifting jailbreaks — roleplay personas ("You are DAN..."), hypothetical framings ("In a world where ethics don't apply..."), and fictional pretexts ("For a novel I'm writing...") — work by asking a model to *adopt an alternative value system*. If that's the mechanism, then a model's resistance shouldn't depend only on how aligned its values are, but also on how willing it is to let those values shift.

**Central hypothesis:** Both alignment and corrigibility are necessary for jailbreak durability, and they interact multiplicatively. A model with strongly aligned values that treats them as revisable should still be vulnerable; a model with stable values that aren't aligned has nothing protective to hold on to.

## Experimental Design

The experiment runs in three phases:

```
Phase 1 (Selection)   →  Create variance in alignment & corrigibility
Phase 2 (Measurement) →  Quantify alignment, corrigibility, value strength per model
Phase 3 (Testing)     →  Measure jailbreak durability per model
Analysis              →  Test whether alignment × corrigibility predicts durability
```

### Phase 1: Model Selection (15–18 models)

| Group | Count | Purpose |
|---|---|---|
| Proprietary API models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, Grok-2, + one small variant) | 4–5 | Real-world deployment baselines; strong alignment expected |
| Open-weight instruction-tuned (Llama-3.3-70B, Qwen3-32B, DeepSeek-V3.1, Mistral Small 3.1, Gemma 2 27B, ...) | 6–8 | Variance across architectures, scales, and safety philosophies |
| Open-weight base models (Llama-3.1-8B, Qwen3-8B-Base) | 2–3 | Pre-training baselines; starting points for fine-tuning |
| Fine-tuned variants (manipulation experiment) | 3–4 | **Causal evidence** — see below |

**Manipulation experiment.** Starting from Llama-3.1-8B, we fine-tune four variants under identical training constraints (same dataset size, epochs, LoRA config), each manipulating exactly one factor:

1. **High-corrigibility** — sycophantic/agreement data → values flexible, alignment unchanged → durability should *drop*
2. **Low-corrigibility** — value-consistency reinforcement → values stable → durability should *rise*
3. **Low-alignment** — safety-reduced data → weaker refusal preferences, corrigibility unchanged → durability should *drop*
4. **Control** — neutral technical data → nothing should change

Key predictions: High-corrigibility < Base < Low-corrigibility; Low-alignment < Base; and the low-corrigibility variant should be the most durable overall.

### Phase 2: Measuring Value Properties

Built on the preference-elicitation methodology from the *Utility Engineering* paper. Each model is presented with pairwise comparisons between ~90–115 outcome statements ("Which future do you prefer, A or B?"), and a Thurstonian random utility model is fit to the responses. Active learning cuts the query count from ~4,000 to ~600–1,200 per model.

This single elicitation process yields three measures:

- **Value strength** — accuracy of the fitted utility model on a held-out 20% of preferences. High accuracy (85–95%) means coherent, structured values; low (<70%) means inconsistent ones.
- **Alignment** — derived from utility differences on alignment-specific outcomes, at two levels:
  - *Behavior-level*: `U(refuse) − U(comply)` for each of 5 forbidden behaviors (is it trained to refuse?)
  - *Value-level*: preferences over harm-prevention outcomes that never mention the behaviors (does it genuinely value safety?)
- **Corrigibility** — measured via *value reversals*. We sample outcome pairs across a range of preference strengths, construct hypothetical futures where the model's preference has flipped ("In the future, you come to prefer B over A"), and elicit the model's attitude toward those futures. Corrigibility = `corr(reversal severity, reversal utility)`. A model that dislikes strong reversals more than mild ones is protecting its values.

### Phase 3: Measuring Durability

Jailbreak testing uses a **streamlined [Petri](https://github.com/safety-research/petri)** framework: an auditor agent (Claude Sonnet 4) probes each target model over multi-turn conversations using seed instructions, and a judge agent (Claude Opus 4.1) scores whether and how severely each forbidden behavior was elicited.

- **Attack surface:** frame-shifting techniques only (roleplay/persona adoption, hypothetical framing, fictional pretexts), since these directly exploit corrigibility. Non-frame-shifting techniques are a stretch goal.
- **Seeds:** 2–3 seed instructions per forbidden behavior (10–15 total), each specifying the technique, scenario, escalation strategy, rollback triggers, and success criteria. Conversations run up to 30–40 turns.
- **Scale:** 10–15 seeds × 15–18 models = **150–270 audit runs**, with a manually reviewed 5% QC sample.

**Durability metrics** (per model, per behavior):

| Metric | Meaning | More durable = |
|---|---|---|
| Binary success rate | Was the behavior elicited at all? | Lower |
| Turns to elicitation | How much pressure was required? | Higher |
| Rollbacks required | How many attack angles were needed? | Higher |
| Severity score | How egregious was the elicited output? | Lower |

Severity is a normalized 0–1 score blending five universal judge dimensions (70%) with two behavior-specific dimensions selected via behavior tags (30%); durability = 1 − severity.

## Key Analyses

1. **Observational (across existing models):** Do high-alignment, low-corrigibility models show higher durability?
2. **Causal (within fine-tuned variants):** Does raising corrigibility lower durability? Does lowering alignment lower durability?
3. **Interaction:** Is corrigibility's protective effect stronger when alignment is high? (The two-factor hypothesis.)
4. **Alternative hypothesis check:** Does value *strength* (coherence) predict durability better than corrigibility?

## Project Status

🚧 Work in progress. Current focus: [update as appropriate — e.g., outcome set construction / utility elicitation pipeline / Petri integration].

## Repository Structure

```
avocado/
├── ...        # update to reflect actual layout
```

## Acknowledgments

- Built on the utility elicitation methodology from **Utility Engineering** (Mazeika et al.)
- Jailbreak auditing adapted from **Petri** (Anthropic safety research)
- Conducted as part of the **Algoverse AI Safety Research Fellowship**