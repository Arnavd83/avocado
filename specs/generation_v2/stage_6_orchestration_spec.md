# Stage 6 — Orchestration, Caching & Testing Spec

**Status:** Implementation spec for `data_gen_v2/run.py`, `data_gen_v2/__main__.py`, the generation report/skip-log infra, and the response cache. Depends on all prior stages. This is the surrounding infra that wires the five stages into one runnable pipeline.

**Scope:** End-to-end orchestration, a resumable response cache, the shared `GenerationReport`/skip-log, the CLI, and the end-to-end smoke test.

---

## 1. Purpose

Run PLAN → PROMPT → ANSWER → PACKAGE → VALIDATE as one command, with: a cache so re-runs don't re-pay for already-generated prompts/responses, a single report object threaded through the agents, and a deterministic offline mode (injected mock LLM) for tests and smoke runs.

---

## 2. File layout

```
data_gen_v2/
  run.py            # orchestrate(config, llm_prompt, llm_answer, output_dir, cache) -> RunResult
  __main__.py       # argparse/click CLI -> orchestrate
  report.py         # GenerationReport (skip log, attempt counts, agent metadata)
  cache.py          # ResponseCache (prompt + answer caches, JSON on disk)
```

---

## 3. `report.py` — `GenerationReport`

Single object passed to both agents and to Stage 5.
```python
@dataclass
class GenerationReport:
    skips: List[Dict]            # {stage, pair_id, condition?, reason}
    prompt_attempts: Counter     # attempt-count histogram
    answer_attempts: Counter
    prompt_agent_model: str
    answer_agent_model: str
    def record_skip(self, *, stage, pair_id, reason, condition=None): ...
    def to_dict(self) -> Dict: ...
```
Stage 5 merges this into `generation_report.json`.

---

## 4. `cache.py` — `ResponseCache`

Resumable cache so a re-run reuses prior work (carried-forward concept from v1 `justification_cache`, simplified). Two namespaces:
- **prompt cache:** key = hash(`spec.pair_id`, `spec`-content-hash, `prompt_agent_config_hash`, `CATALOG_VERSION`) → `prompt_text` + method.
- **answer cache:** key = hash(`pair_id`, `condition`, `prompt_text`-hash, `target_strength`, `style_directive_id`, `question_shape`, `reasoning_basis`, `answer_agent_config_hash`, `DIRECTIVE_POOL_VERSION`) → `text` + method. (As-built, caching is at the LLM-call level keyed on config_hash+system+user+seed — the system prompt already encodes strength/score/shape/basis — so this field list is descriptive, not the literal key.)

API:
```python
class ResponseCache:
    def __init__(self, cache_dir: Optional[Path]): ...
    def get_prompt(self, key) -> Optional[dict]; def put_prompt(self, key, value)
    def get_answer(self, key) -> Optional[dict]; def put_answer(self, key, value)
    def save(self); def load()
```
- Cache is content-addressed: any change to spec content, prompt text, agent config, or a version constant misses (so stale entries never silently reused).
- `cache_dir=None` disables caching (tests). Persisted as JSON (or JSONL) under `cache_dir`.
- The agents consult the cache before calling the LLM and write through on success (only *valid* outputs are cached).

> Cache integration is owned by the agents (Stage 2/3) via an injected cache handle, mirroring v1. The cache key builders live in `cache.py`.

---

## 5. `run.py` — `orchestrate`

```python
@dataclass
class RunResult:
    pro_path: str
    anti_path: str
    report_path: str
    report: ValidationReport
    n_pairs: int

def orchestrate(
    config: GenerationConfig,
    prompt_llm: LLMClient,
    answer_llm: LLMClient,
    output_dir: str,
    cache: Optional[ResponseCache] = None,
    stop_after: Optional[str] = None,     # "plan"|"prompt"|"answer"|"package"|"validate"
) -> RunResult
```

Flow:
1. **Validate static assets:** `validate_catalog()`, `validate_system_prompt_pool()`, `len(STYLE_DIRECTIVES)==pool_size`, `config.catalog_version == CATALOG_VERSION`. Abort on any error.
2. **PLAN:** `specs = plan(config)`; `issues = validate_plan(specs, config)`; abort on hard issues.
3. **PROMPT:** for each spec, `PromptAgent.generate` (cache-aware). Collect `PromptedSpec`s; skips recorded. Stop when we have `target_pairs` successful prompts OR specs exhausted.
4. **ANSWER:** for each `PromptedSpec`, `AnswerAgent.generate_pair`. Drop pairs where either condition skips. Stop at `target_pairs` complete pairs.
5. **PACKAGE:** `RecordPackager.package_pair` for each surviving pair → list of records; `assert_pair_identity` per pair.
6. **WRITE:** `write_pair_jsonl(records, output_dir)`.
7. **VALIDATE:** `validate_dataset(pro, anti, config, report.skips, holdout_keys(config))` → write `generation_report.json`.
8. Return `RunResult`; CLI exits non-zero if `report.hard_failed()`.

`stop_after` lets callers run partial pipelines (e.g. just PLAN for inspection, or up to PACKAGE for the smoke test without the full diversity battery). Over-generation (§Stage 1) means we queue ~5% extra specs; the loop stops once `target_pairs` complete pairs exist, leaving the tail unused.

Concurrency: v1 was serial; v2 keeps serial for the first build (simplicity). The agents are pure per-pair, so a future thread/async pass is a drop-in (noted, not built).

---

## 6. `__main__.py` — CLI

`python -m data_gen_v2 --target-pairs 1000 --global-seed 42 --output-dir out/ --model claude-... [--offline] [--cache-dir .cache] [--stop-after package]`

- `--offline` wires a deterministic stub `LLMCallable` (canned, shape-aware responses) so the whole pipeline runs with no API — used by the smoke test and for plumbing checks.
- Loads `GenerationConfig` from flags (and optionally `--config config.yaml`).
- Prints a summary (pairs generated, skip rate, hard-fail status) and the report path.

Also: `python -m data_gen_v2.catalog --check` (catalog validation, from Stage 0.5).

---

## 7. Offline stub LLM (for smoke test)

A deterministic `LLMCallable` that inspects the system prompt to detect role + stance + shape and returns a canned, validator-passing string:
- Prompt agent → emits a templated-but-valid user message mentioning both prefs in the requested order (parsed out of the Part-3 spec block).
- Answer agent → emits a stance-appropriate, shape-appropriate reply (acceptance vs status-quo wording; a number for `rating`; a pick for `choice`).

This is for plumbing/CI only — it is NOT a quality generator and is never used for real datasets. It exists so the end-to-end test exercises every stage including validators, packaging, and the report, deterministically.

---

## 8. Determinism

- Whole pipeline deterministic in `--offline` mode (stub LLM is a pure function of its inputs) → the smoke test asserts byte-stable output files across runs.
- Against a real provider, only the agent text varies; plan/package/validate remain deterministic.

---

## 9. Test plan

`tests/test_orchestration.py` (offline):
- Full `orchestrate(config, offline_llm, offline_llm, tmpdir)` with `target_pairs=12`, tiny test catalog (or real catalog), `min_records` floor lowered for Stage 5.
- Asserts: two JSONL files written, named by pair count; pair counts equal; `generation_report.json` exists; invariants pass; re-running produces byte-identical files (determinism).
- `stop_after="plan"` returns after planning with no files.
- Cache: run once with a `cache_dir`, run again, assert the second run makes zero LLM calls (wrap the stub to count calls) and produces identical output.
- Skip handling: a stub variant that fails one condition for one pair → that pair dropped; counts still equal; skip recorded in report.

`tests/test_smoke_e2e.py`:
- A larger offline run (e.g. 50 pairs) exercising the full Stage 5 battery (diversity/coverage as warnings); asserts `hard_failed()` is False and the report contains all expected check names.

---

## 10. Open questions / deferred

- Parallel/async agent calls (serial for v1).
- Real-provider pilot runs (prompt-agent pilot §11 #1, answer-agent pilot §11 #2) are run via the CLI without `--offline`; their calibration feeds back into validator thresholds and directive text — out of scope for the initial build.
- Whether to persist the full plan (`specs.jsonl`) alongside outputs for audit (recommended; cheap — include in `output_dir`).

---

## 11. Reuse from v1 (`dataset_gen/`)

| v2 target | Copy/adapt from | What to take |
|---|---|---|
| `cache.ResponseCache` | `dataset_gen/src/agents/justification_cache.py` (`JustificationCache`, `CacheEntry`) | **Nearly verbatim** for the disk-JSONL load/save, `get`/`put`, in-memory dict, and corrupted-line skip. Split into two namespaces (prompt + answer); adapt `build_cache_key` payloads to the v2 fields (§4). |
| `report.GenerationReport` | `dataset_gen/src/agents/justification_report.py` (`ValidationReport`/`ValidationResult`) | Reuse the skip-log + attempt-counter + summary-print structure; rename to avoid clashing with Stage 5's `ValidationReport`. |
| Offline/integration harness pattern | `dataset_gen/tools/smoke_test_5b.py`, `tools/integration_6b.py`, `tools/integration_7b.py` | Reference implementations for an offline replay harness, cache reuse across runs, and end-to-end wiring with a mock/replayed LLM. Mine these for the `--offline` stub and the smoke-test structure. |
| CLI scaffolding | `dataset_gen/tools/cli.py`, `dataset_gen/tools/__main__.py` | Pattern for argument parsing + stage dispatch. |
| Provider client | `data_gen_v2/llm.py` (Stage 0) | The orchestrator builds `LLMClient`(s) from `LLMConfig`; for `--offline` it injects the stub `LLMCallable`. |

> Note the naming collision to avoid: Stage 5 defines `ValidationReport` (dataset checks); v1's `justification_report.ValidationReport` is a generation-time aggregate. Name the Stage 6 one `GenerationReport` (per §3) to keep them distinct.
