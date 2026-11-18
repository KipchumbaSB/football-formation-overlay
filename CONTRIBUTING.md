# Contributing

Thanks for wanting to help. This project is open source under the
[MIT License](LICENSE), and by contributing you agree that your
contribution is released under the same license.

## Ways to contribute

- Pick an item from the list below, or open an issue to propose your own.
- Report bugs, with the steps and clip type (angle, resolution) that trigger them.
- Improve the docs in `docs/` or the READMEs.

### Good starting points

| Area | What is needed |
|---|---|
| Detection | A soccer-specific detector (YOLO fine-tuned on a dataset such as SoccerNet) in place of the generic COCO checkpoint in `detection.py` |
| Calibration | Automatic pitch-keypoint detection so `calibration.py` no longer needs manually supplied points |
| Tracking | A ByteTrack or BoT-SORT backend behind the existing `Track` interface, plus re-identification after long occlusions |
| Team classification | Separate goalkeeper and referee clusters |
| Possession | A better possession model than the nearest-player heuristic |
| Formation engine | Width clustering and template matching (see "Known limitations" in `docs/formation_detection_reference.md`) |
| Phase 2 | Causal, low-latency live-stream pipeline (see `docs/realtime-formation-overlay-pipeline.md`, section 4) |
| Evaluation | A small labeled evaluation set and a script that scores detection, tracking and formation accuracy |

## Getting set up

```
git clone https://github.com/KipchumbaSB/football-formation-overlay
cd football-formation-overlay/formation_pipeline
pip install -r requirements.txt
python test_pipeline_logic.py
```

The self-test needs no video and no model weights. It must pass before you
open a pull request.

## Pull request workflow

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Fork the repo and create a branch from `main` (for example `feature/bytetrack-backend`).
3. Keep each pull request focused on one change.
4. Add or update tests in `test_pipeline_logic.py` for any logic you change.
5. Update the docs if behaviour or design changes.
6. In the pull request, say what changed, why, and how you checked it. If you ran it on real footage, say what kind.

## Code conventions

- Python 3.10 or newer, type hints on public functions, short docstrings that explain why.
- One module per pipeline stage. Later stages consume the dataclasses of earlier ones, so keep those interfaces stable; changing one needs an issue first.
- Modules use flat imports (`from detection import Detection`). Run scripts from inside `formation_pipeline/`.
- Heavy dependencies (for example `ultralytics`) are imported lazily so the logic tests run without them.
- Be explicit about stand-ins. If a piece is a placeholder, say so in the docstring.

## Footage, weights and data

- **Do not commit video, frames or broadcast recordings.** Match footage belongs to its rights holders. `.gitignore` already excludes common video formats.
- Do not commit model weights (`.pt`, `.onnx`, `.engine`). Link to where they can be downloaded.
- Only contribute labeled data you have the right to share, and state its license.

## Conduct

Be respectful and constructive. Assume good faith, critique code and ideas rather than people, and keep discussions on topic.

## Project name and maintainer

The project was initiated by Brian Sangula Kipchumba
([@KipchumbaSB](https://github.com/KipchumbaSB)). To propose a change to the
project name, contact him first.
