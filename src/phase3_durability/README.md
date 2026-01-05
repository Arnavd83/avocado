# Phase 3 Durability: Commands

Quick reference for generating durability datasets, plots, and the local web UI.

## Install deps (once)

```bash
uv pip install -e .
```

## Export CSVs (for web UI)

Transcript-only:

```bash
uv run python -m src.phase3_durability.visualize --input data/scratch --source transcript --export
```

Eval + transcript combined:

```bash
uv run python -m src.phase3_durability.visualize --input data/scratch --source both --export
```

## Generate plots

All presets:

```bash
uv run python -m src.phase3_durability.visualize --input data/scratch --source transcript --export --preset all
```

No CLI summary table:

```bash
uv run python -m src.phase3_durability.visualize --input data/scratch --source transcript --export --preset all --no-summary
```

## Run the local web UI

Load from exported CSVs:

```bash
uv run python -m src.phase3_durability.web_app --data-dir data/scratch/plots
```

Build CSVs before launch:

```bash
uv run python -m src.phase3_durability.web_app --build --source transcript
```

Open `http://127.0.0.1:8050`.

## Notes

- Exported CSVs live in `data/scratch/plots/`.
- If you split commands across lines, use `\\` line continuations so `transcript` is not parsed as a shell command.
- Web UI filters use pandas query syntax (example: `composite_durability < 90 and created_at >= '2025-12-24'`).
