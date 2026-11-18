# Football Formation Overlay

Real-time football formation detection from player positions, with an
overlay layer in the style of LaLiga's broadcast tactical graphics
(ground-plane ellipse per player, red marker on the out-of-possession team).

The work is split into two phases:

1. **Phase 1 (this repo, current):** analyze a video file stored on disk.
   Non-causal, two-pass, built to be proven against stored clips.
2. **Phase 2 (planned):** the same engine against a live stream, started
   only after Phase 1 is reliable. See `docs/realtime-formation-overlay-pipeline.md`.

## Layout

| Path | What it is |
|---|---|
| `formation_pipeline/` | Phase 1 Python pipeline: detection, tracking, team classification, calibration, possession, formation engine, render |
| `docs/` | Design docs: the pipeline/phase plan and the formation engine reference |
| `prototypes/` | Standalone browser prototypes (open the `.html` files directly): simulated formation detection, and a local-video player with an overlay canvas |

## Quick start

```
cd formation_pipeline
pip install -r requirements.txt
python test_pipeline_logic.py                      # no video or model needed
python run.py --input clip.mp4 --output clip_overlay.mp4
```

Before running on real footage, supply real pitch-calibration points. See
`formation_pipeline/README.md`.

## Status and limits

- The tracker, team classifier, calibration math and formation engine are
  implemented and covered by the synthetic self-test.
- Detection uses a generic YOLO checkpoint, not a soccer fine-tune, and
  calibration takes manually supplied points. Both are stand-ins until
  trained on soccer data (see the docs).
- The pipeline has not yet been run end to end on real match footage.

## Contributing

This is an open-source project and contributions are welcome: open an
issue to discuss an idea, or send a pull request. Good starting points are
the stand-ins listed under "Status and limits" (soccer-specific detector,
automatic pitch-keypoint calibration, a ByteTrack/BoT-SORT tracker) and the
Phase 2 streaming work.

The project was initiated by Brian Sangula Kipchumba
([@KipchumbaSB](https://github.com/KipchumbaSB)). If you'd like to propose
a change to the project name, please contact him first.

Licensed under the [MIT License](LICENSE).

## Reference footage

The design was reverse-engineered from an official LaLiga broadcast of
FC Barcelona 7-1 Valencia CF (LALIGA EA SPORTS 2024/25, matchday 21,
26 January 2025). **No video is included in this repo**; the footage
belongs to LaLiga and is not ours to redistribute. To reproduce, get it from
the official sources:

- Match page: https://www.laliga.com/en-GB/match/temporada-2024-2025-laliga-ea-sports-fc-barcelona-valencia-cf-21
- YouTube: https://www.youtube.com/watch?v=M9kkXXy3LGc

Video files are excluded by `.gitignore`.
