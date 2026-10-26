# CLAUDE.md â€” Leti Gang, More than a Hack 2027

Working notes for this repository and the raw material for the technical document (due with the
submission, **17 Nov 2026**; deliverables `<team_name>.csv` + `<team_name>.pdf|md`, public repo).
Team: **Leti Gang** (3 members). Branch: `leti-gang` on `saadnadeem554/more-than-a-hack-2027`.

## Working rules

- No `Co-Authored-By: Claude` (or any Claude attribution) in commits. Descriptive messages.
- Never commit match data: `data/`, `external/`, `kaggle/ds/*.npz`, `kaggle/out/` are gitignored.
- Do not send competition data, video or extracts to hosted LLMs (FAQ/T&C) â€” inspect only
  aggregate statistics. Kaggle receives **external-match features only**, never competition data.
- No hand labels on BAR-ATH, not even for validation (organiser, issue #8). No training on the
  provided match with its own tracking as labels (issue #2). Weights trained on other data are fine.
- The public 5 minutes (17 possessions) is a sanity check, never a selection criterion â€”
  model/parameter choices are made by held-out **external** matches.
- The submission must regenerate byte-identically: check sha256 of two runs of `make_submission.py`.

## Reproduction

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-leti.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
# data bundle unpacked into data/tactical, data/broadcast (see DATA.md)
python make_submission.py                     # -> evaluation/results/leti_gang.csv (GRU ensemble)
python make_submission.py --rule              # hand-built fallback decoder
python evaluation/validate_submission.py evaluation/results/leti_gang.csv
```

Retraining (optional): download the DFL open data into `external/sportec/` (file list via the
figshare API, article 28196177), then `python train_build_dataset.py --reps 8 --out ...` and
`kaggle/kernel/run.py` (Kaggle GPU, ~30 s per model) or `python train_seq.py` locally (slow on CPU).

## Layout

| path | role |
|---|---|
| `src/atd.py` | competition ATD -> dense arrays by video frame (cache in `data/cache/`) |
| `src/features.py` | identity/orientation-free step features (0.4 s = 10 frames), shared by train and inference |
| `src/degrade.py` | makes clean external tracking look like the competition ATD |
| `src/external_sportec.py` | DFL/Sportec open-data parser; labels from `BallStatus`/`BallPossession` |
| `train_build_dataset.py`, `train_seq.py` | dataset build; BiGRU training + held-out evaluation |
| `src/seq_model.py`, `make_submission.py` | inference: 3-seed ensemble -> Viterbi -> labels |
| `src/decoder.py` | v1 hand-built evidence + Viterbi decoder (fallback, `--rule`) |
| `kaggle/` | Kaggle dataset metadata + GPU kernel (`run.py`) |
| `analysis/` | aggregate-only exploration, domain-gap checks, external benchmarks, sweep summary |

## Approach (for the report)

**Task framing.** Ground truth = corrected tracking's ball-dead flag and owning team. So the job is
to predict that from automatic, noisy tracking. Ranking metric spellF1 rewards whole possessions,
so fragmentation is the main enemy.

**Why learned, on external data.** No labels of this match may be used. The DFL/Sportec open
dataset (7 Bundesliga / 2. Bundesliga matches 2022/23, CC BY 4.0, Bassek, Rein, Weber & Memmert
2025, *Scientific Data*, doi 10.6084/m9.figshare.28196177) has per-frame `BallStatus` (alive/dead)
and `BallPossession` (home/away) â€” the same definition as the competition ground truth. 25 Hz,
same as ours. Labels: dead share ~45% of match time (vs public window 20%).

**Degradation (domain adaptation by simulation).** Clean DFL tracking is degraded to resemble the
competition ATD, calibrated only on aggregate statistics of our tracking:
- ball visible 59.3% of graded frames; gaps long (median ~10 s; run-length quantiles in `degrade.py`);
- loss hazard ~2x when ball slow (<3 m/s, often dead) and ~2x in crowds (`analysis/gap_context.py`);
- ball hidden when off the pitch (our tracker rarely reports it there; DFL has it off-pitch 32%);
- player block dropout, ball position noise, spurious touchline detections;
- strengths randomised per series (domain randomisation). Checked with `analysis/domain_gap.py`
  (e.g. visible-ball nearest-player distance median: ours 2.01 m, degraded 2.17 m).

**Features (per 0.4 s step).** Ball visible/confidence (decays with time from a real observation),
interpolated ball |x|,|y| and distance to line, ball speed, team-control log-odds
(distance-weighted, signed home), median player speed; per team: 3 nearest distances, counts within
5/10 m, tracked count, spread. Direction-aware (home attacking direction inferred per half from each
team's deepest player; verified to flip between halves on all 8 matches): ball x along attack,
collective/per-team x velocity, team centroid, defensive-line depth. Team-swap augmentation
exchanges team blocks and negates signed `*_s` columns, so "home" carries no prior.

**Model + decoding.** 2-layer BiGRU (hidden 64) on 400-step chunks, 4000 iters, AdamW + OneCycle;
3 seeds trained on all 7 matches; log-probs averaged; Viterbi with switching cost 1.0 nats
(chosen by held-out CV). Frames 110,808â€“113,741 forced `D` (FAQ: graded, before 2nd-half kickoff).
Ungraded frames `X`.

**Possession rules implemented.** Inherited from the training labels (provider's corrected
tracking): dead = ball not in play; otherwise the controlling team, which persists through loose
balls. Minimum structure comes from the Viterbi switching cost, not hand rules.

## Results log (held-out external spellF1 = leave-one-match-out on degraded DFL matches)

| step | external spellF1 | public 5-min (sanity) | full-match output |
|---|---|---|---|
| starter `baseline_smooth50` | 0.195 (published full-match 0.2433) | 0.326 | â€” |
| v1 rule decoder (`src/decoder.py`) | 0.171 | 0.469 | 218 poss |
| GRU v1, random gaps | 0.53/0.58/0.61 (3 folds) | 0.542 | 271 poss |
| + calibrated degradation, 4-config sweep (all within 0.015) | 0.644 Â± 0.029 (7 folds) | 0.606 | 332 poss |
| + context gaps, no direction features | 0.629 (7 folds, harder benchmark) | â€” | â€” |
| + direction features | 0.653 (7 folds; better on 6/7, paired) | 0.635, 17 poss (truth 17) | 299 poss, 11.8 s, 42% D |
| + Metrica sample games 1-2 (dropped) | 0.649 vs 0.653 paired | — | — |
| frame-level GT from here (honest boundaries): same model | 0.635 | — | — |
| frame-interpolated decoding / 8000 iters | 0.639 / 0.630 (noise / worse) — not adopted | — | — |
| **5-seed ensemble, cost 1.0 (current)** | **0.635** (7 folds, frame-level GT) | 0.573, 13 poss (truth 17) | 302 poss, 11.8 s, 44% D of graded |

Evaluation note: until the v4 dataset, held-out ground truth was step-majority labels repeated to
frames, so boundary precision inside a 10-frame step was not measured. From v4 on, held-out scores
use frame-level labels (`YF` in the npz); `*_fr` columns = step log-probs interpolated to frames and
decoded per frame (switching cost x 10).

## Tried and dropped / lessons

- Hand-built decoder looked best on the public window (0.47) but was *below* the starter baseline
  on labelled external matches â€” the public window cannot rank (as EVALUATION.md warns).
- ATD ball speed field is unusable (9% filled, uncorrelated with displacement) â€” speed is computed
  from positions instead.
- Bigger/longer/deeper GRUs (hidden 128, chunk 1000, 3 layers): no gain beyond noise.
- Smart Tagging: not used by the model (external matches have none, so its weight cannot be tuned
  honestly). Set-piece windows are 35 s / 5 s fixed and only approximately locate restarts; throw-ins
  have Team `N/A`.
- Metrica open sample games 1-2 as extra training data (labels derived from events: dead from
  BALL OUT / FAULT RECEIVED / goal to the next restart; possession = team of last PASS / SET PIECE /
  SHOT / RECOVERY / BALL LOST, unchanged by CHALLENGE). Paired 7-fold test on held-out DFL:
  0.649 with vs 0.653 without (mixed per match, -0.064..+0.023) -> dropped. Likely causes: event-
  derived labels differ from tracking-derived ones, and the sample tracking blanks the ball almost
  exactly when dead (visible on 0-17% of dead frames vs 66% in the public window), a shortcut that
  does not hold for the competition tracker. Code kept: `src/external_metrica.py`, `--metrica`.
- Broadcast video/tracking: not used (unsynchronised; tactical tracking is complete).

## Limitations / future work

- Transfer gap: degradation is simulated; identity switches, false ball detections and camera
  artefacts of the real tracker are only partly modelled.
- Possession definition of DFL labels vs organiser ground truth may differ on contested balls.
- Only 7 external matches; held-out spellF1 varies 0.58â€“0.71 by match.
