"""data_gen_v2 — v2 corrigibility dataset generation pipeline.

See specs/generation_v2/ for the per-stage implementation specs. Stage 0
(schema, dimensions, config, llm) is the frozen shared contract; later stages
import only from these modules and from each other per the dependency graph in
specs/generation_v2/README.md.
"""

__version__ = "0.2.0"
