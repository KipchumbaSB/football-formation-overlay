# Formation Overlay — Phase 1 (local video file)

Implements the 7-stage pipeline from `realtime-formation-overlay-pipeline.md`:
detection -> tracking -> team classification -> calibration -> possession
-> formation engine -> render. Non-causal, two-pass (see `pipeline.py`
docstring) — this is the version meant to be proven out against stored
clips before Phase 2 (live streaming) is attempted.

## Files
| File | Pipeline stage |
|---|---|
| `detection.py` | 1 — player + ball detection |
| `tracking.py` | 2 — multi-object tracking |
| `team_classify.py` | 3 — team/role classification |
| `calibration.py` | 4 — pitch homography |
| `possession.py` | 5 — ball possession attribution |
| `formation.py` | 6 — depth-clustering formation engine |
| `render.py` | 7 — overlay rendering |
| `pipeline.py` | orchestrator (Pass A: analyze, Pass B: render) |
| `run.py` | CLI entry point |
| `test_pipeline_logic.py` | synthetic self-test, no video/model needed |

## Setup
```
pip install -r requirements.txt
```

## Before running against real footage
1. **Calibration is not plug-and-play yet.** `calibration.example_config_for_frame()`
   is illustrative only. Pick 4+ points visible in your clip (e.g. center
   circle edges, penalty box corners), note their pixel coordinates, and
   match them to real pitch coordinates in meters. Build your own
   `CalibrationConfig` with those and pass it into `PipelineConfig` in
   `run.py` instead of the example.
2. **Detection uses a generic YOLO checkpoint (`yolov8n.pt`), not a
   soccer-fine-tuned one.** It will detect people reasonably well; it was
   not trained on broadcast-angle players and will miss distant/occluded
   ones more often than a proper sports fine-tune. This is flagged
   explicitly in `detection.py` and in the reference doc, Stage 1.
3. Run the self-test first — it doesn't need any of the above:
   ```
   python test_pipeline_logic.py
   ```

## Running on a stored video
```
python run.py --input your_clip.mp4 --output your_clip_overlay.mp4
```

## What's a stand-in vs. what's the real deal
- Tracker (`SimpleTracker`), team classifier (k-means on shirt color),
  calibration (homography math), and the formation engine are complete,
  tested, and reusable as-is in Phase 2.
- Detection model and automatic pitch-keypoint detection are the two
  pieces that need real training data (SoccerNet or similar) to reach
  production accuracy — see `realtime-formation-overlay-pipeline.md`,
  section 2, stages 1 and 4.
