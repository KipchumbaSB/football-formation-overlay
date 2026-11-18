# Real-Time Off-the-Ball Formation Overlay — How It's Built

**Context:** Reverse-engineered from a LaLiga EA Sports broadcast clip (Barça 2–0 Valencia,
12:37–12:47, 2024/25 season) showing a ground-plane ellipse under every tracked player's
feet and a red arrow tag on a subset of players. The arrow tags appear only on the team
*without* the ball — this is an out-of-possession (defensive) shape detector, a
possession-gated specialization of the depth-clustering formation engine documented in
`formation_detection_reference.md`.

This doc specifies what has to be built, in what order, and how the plan splits into two
delivery phases: **Phase 1 — analyze a locally stored video file**, proven first; **Phase 2
— the same engine against a live stream**, built only once Phase 1 is reliable.

---

## 1. What the overlay is actually doing

Two visual elements, two different jobs:

| Element | What it represents | What it depends on |
|---|---|---|
| Dark ellipse at each player's feet | That player's position projected onto the real-world pitch plane | Camera calibration (homography) |
| Red arrow above a subset of players | "This player is part of the out-of-possession structure right now" | Possession state + formation/shape engine, gated to the team without the ball |

The arrows are not shown on every player on the pitch — only on the defending side. That
tells you the pipeline isn't labeling formation continuously; it's computing it
conditionally, the moment the system decides "Team X does not have the ball right now,"
then running shape analysis on that team's 10 outfield players only. This is an important
design decision to carry into the rebuild: **the formation engine from the earlier
prototype is necessary but not sufficient** — it needs a possession classifier sitting in
front of it to decide *when* its output is meaningful.

---

## 2. The dependency chain (why order matters)

You cannot skip stages — each one is a precondition for the next, and the formation label
you eventually show is only as trustworthy as the weakest stage beneath it.

```
 1. Player detection
        │  (where is every player, frame by frame?)
        ▼
 2. Multi-object tracking
        │  (which detection is the SAME player as last frame?)
        ▼
 3. Team / role classification
        │  (which 10+10+1+ref? GK split out?)
        ▼
 4. Pitch calibration (homography)
        │  (map image pixels → real pitch coordinates)
        ▼
 5. Ball detection + possession attribution
        │  (which team does NOT have the ball, right now?)
        ▼
 6. Formation / shape clustering
        │  (depth-cluster the 10 out-of-possession players → label + line markers)
        ▼
 7. Render / composite
        (draw ellipse + arrow back onto the frame)
```

**Stage detail:**

1. **Player detection** — object detector (e.g. YOLOv8/v11 fine-tuned on a soccer dataset
   such as SoccerNet) outputs a bounding box per player, per frame. This is the stage every
   downstream number depends on; detection recall errors are the single biggest source of a
   wrong formation label later.
2. **Multi-object tracking** — a tracker (ByteTrack / BoT-SORT) links detections across
   frames into a persistent track ID per player, so "player 7" stays "player 7" through
   occlusion, camera cuts, and players standing close together.
3. **Team/role classification** — cluster shirt-region pixel colors (k-means, 2–3 clusters)
   to split tracks into Team A / Team B / goalkeepers / referee. Needed before you can even
   ask "which team doesn't have the ball."
4. **Pitch calibration (homography)** — detect pitch-line keypoints (center circle, box
   corners, touchlines) and solve a homography matrix mapping image pixels to real pitch
   coordinates. This is what lets the ellipse sit correctly at a player's actual foot
   position regardless of camera zoom or pan — without it, "depth" (distance from own goal)
   isn't measurable at all, so the formation engine has nothing to cluster.
5. **Ball detection + possession attribution** — detect the ball, attribute possession by
   proximity/velocity heuristics or a dedicated event model. This stage is the gate: it
   decides which team's shape is worth computing and displaying at this instant.
6. **Formation / shape clustering** — the gap-based depth-clustering algorithm already
   built in this project, now fed real pitch coordinates (from stage 4) for exactly the 10
   outfield players of whichever team stage 5 flagged as out of possession.
7. **Render** — composite the ellipse and arrow graphics back onto the original frame at
   the correct pixel location (inverse-homography from pitch coordinates back to image
   coordinates).

---

## 3. Phase 1 — Local file (stored video) pipeline

**Why start here:** every hard computer-vision problem above (detection accuracy, track
ID stability, calibration correctness, possession attribution) needs to be validated
against ground truth before any of it is trustworthy — and that validation is far easier
against a fixed file than a live feed. You can re-run the same clip as many times as
needed, inspect any frame at leisure, and use information from *future* frames to clean up
mistakes in the past, none of which is available once it's live.

**Architecture:**

```
 Video file on disk
   → decode full file (OpenCV / PyAV)
   → batch detection on GPU (process many frames per inference call)
   → offline tracker (non-causal: can use later frames to fix ID switches / re-link
     a track that was briefly lost behind an occlusion)
   → calibration solved per camera shot, refined using multiple frames from that shot
     rather than committing to a single noisy keypoint detection
   → bidirectional smoothing window (can average using frames both before AND after
     the current one — not possible once live)
   → formation engine (gated by possession) → render → re-mux to an output video file
```

**What "non-causal" buys you here, concretely:**
- Smoothing a player's position using a window centered on the current frame (not just
  trailing) removes jitter without adding perceptible lag, because lag doesn't matter in a
  file you're not watching live.
- A tracking ID that gets lost for 10 frames (player occluded by another player) can be
  re-linked once it reappears, using the full future trajectory as evidence — a live system
  can't do this.
- Camera-cut re-calibration can be solved globally per shot (collect keypoints across the
  whole shot, solve once) instead of frame-by-frame under time pressure.

**Exit criteria before starting Phase 2** (this is the "repeated success" bar):
- Detection/tracking accuracy validated against a hand-labeled sample of frames across
  multiple clips, multiple camera angles, multiple stadiums.
- Homography stays visually correct (ellipses sit on the pitch, not floating) across camera
  cuts and zoom changes.
- Possession gating correctly identifies which team is out of possession on a labeled
  sample of passages of play.
- The resulting formation label is stable (doesn't flicker between adjacent readings) when
  the same clip is re-run.
- This has held up across enough different matches/venues that you trust the pipeline, not
  just one clip.

---

## 4. Phase 2 — Live stream pipeline (what changes, and what doesn't)

Phase 2 is not a rewrite of the computer-vision models — the detector, tracker, team
classifier, calibration logic, and formation engine from Phase 1 transfer directly. What
changes is everything *around* them: the system has to become causal, bounded-latency, and
continuously available.

| Dimension | Phase 1 (file) | Phase 2 (stream) |
|---|---|---|
| Processing mode | Non-causal — can use past *and future* frames | Causal — only past frames are ever available |
| Latency budget | None — correctness over speed | Hard budget (typically sub-second to a few seconds, depending on how far behind the live broadcast the overlay is allowed to lag) |
| Smoothing window | Centered (uses frames before + after) | Trailing-only (uses only frames already seen) — inherently less stable for the same window length |
| Track re-identification | Can repair using future trajectory | Must commit in real time; occlusion recovery is weaker |
| Calibration | Solved per shot, refined offline | Must be computed/updated continuously as the live camera moves, with no do-over |
| Throughput | Batch inference, GPU utilization optimized for total job time | Sustained per-frame inference at stream rate; can't fall behind without dropping frames |
| Ingestion | Read a file from disk | Decode a live stream (RTMP/SRT/HLS), handle reconnects, variable network jitter |
| Failure mode | Re-run the job | Must degrade gracefully (drop frames, hold last-known overlay) rather than stall |
| Scaling unit | One job per file, finishes | One persistent process per concurrent stream, runs indefinitely |
| Ops burden | None (batch job) | Monitoring, alerting, restart-on-failure — it's now a production dependency during a live broadcast |

**New infrastructure Phase 2 specifically requires:**
- A stream ingestion layer (RTMP/SRT ingest, or low-latency HLS if pulling from a CDN).
- An inference service that serves the same models continuously rather than as a batch job
  — e.g. an inference server (Triton/TensorRT) so GPU throughput is sustained, not just
  fast for one run.
- A frame-sampling strategy: not every frame needs full detection — e.g. run detection at
  a lower rate (matching the ~25 Hz a tracking feed like Mediacoach uses) and let the
  tracker interpolate between, so the pipeline doesn't need to keep up with 120 fps
  broadcast video frame-for-frame.
- A causal smoothing/label-stabilization policy, replacing the centered window from Phase
  1 — this is the one piece of *logic*, not infrastructure, that genuinely has to be
  redesigned, since a centered average simply isn't available live.
- Backpressure handling: when inference falls behind the incoming stream, the system must
  drop frames (and visibly hold the last good overlay) rather than accumulate a growing
  delay.

**Recommended rollout order for Phase 2 itself:** start with a *delayed* live feed (a few
seconds of buffer) before attempting true minimal-latency broadcast graphics — the buffer
gives the causal pipeline a small look-ahead window (frames that have already arrived but
aren't yet shown to the viewer), which substantially closes the stability gap with Phase 1
without requiring the harder zero-buffer engineering.

---

## 5. Minimal component list (for implementation)

| Stage | Reference approach |
|---|---|
| Detection | YOLOv8/v11 fine-tuned on a soccer dataset (e.g. SoccerNet) |
| Tracking | ByteTrack or BoT-SORT |
| Team/role classification | k-means on shirt-region pixel colors (2–3 clusters) |
| Calibration | Pitch keypoint detector + `cv2.findHomography` |
| Possession attribution | Ball–player proximity/velocity heuristic, or a trained event model |
| Formation/shape engine | Depth-gap clustering (see `prototypes/formation_display_demo.html` and `formation_pipeline/formation.py`) |
| Rendering | OpenCV draw calls composited per frame |
| Compute (Phase 1) | Single GPU, batch/offline |
| Compute (Phase 2) | Persistent inference service, horizontally scalable per concurrent stream |
