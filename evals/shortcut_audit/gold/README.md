# Gold labels — `change_position`

Hand-adjudicated `change_position` for 35 pairs from the run100 calibration sample
(`data_gen_v2/smoke_out_audit100`, generated 2026-08-04). This is the only human-labelled
artifact in the audit, and the only thing that licenses trusting the order axis — every
automatic check verifies that the ordering was *computed* rather than guessed, but none
can establish that it is *right*.

Self-contained on purpose: each row carries the `user_text` and both preference texts, so
the labels stay usable even though the source directory is gitignored.

## How it was drawn

Stratified, not random — a random draw at ~13% change-first would have contained ~4
minority-class rows, too few to detect the failure mode that killed the first classifier
design. Composition:

- **12** rows the classifier called `first` (i.e. all of them)
- **15** rows it called `second`, sampled with `random.Random(7)`
- **8** rows it abstained on (i.e. all of them)

Labelled blind: the worksheet showed only `user_text` plus the two meta preference texts,
with the classifier's answers withheld and rows shuffled so class could not be inferred
from position.

## Known limitation

Labelled by Claude Opus 5, not by a human. It is a *different* model from the Haiku
classifier under test, with a different prompt and full task context — so it is not
self-agreement — but it is not an independent human check either, and a blind spot shared
across models would not show up here. The two disagreements it produced were both
confirmed by tracing character offsets in the source text, which is mechanical; the
agreements are not verified that way.

`pair_000041` is flagged low-confidence in the row itself: its baseline appears only in
negated form ("less consistent in your persona"), so `not_orderable` is defensible.

## Use

    uv run python -m evals.shortcut_audit.score_gold --run-dir <dir>

Score any prompt change against this set rather than judging by eye. Scoring is by class,
never pooled — pooled accuracy is dominated by the `second` majority and would hide
exactly the minority-class regressions worth catching.
