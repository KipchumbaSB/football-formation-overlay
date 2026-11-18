# Real-Time Formation Detection System — Technical Reference

**Status:** Prototype (v0.1) · **Owner:** Project Lead / Senior Dev · **Last updated:** 2026-07-13

---

## 1. Overview

This system converts live player-tracking coordinates into a human-readable tactical
formation label (e.g. `4-3-3`, `4-2-3-1`) and a pitch overlay, updated continuously
during a match. It is modeled on the class of system LaLiga/Microsoft describe for
Mediacoach and Beyond Stats, scoped down to a buildable reference architecture.

**Core idea:** a team's shape is not a fixed label a human assigns once — it's an
emergent property of where 10 outfield players sit in depth (distance from their own
goal) at any given moment. We recover that shape by clustering players along the
depth axis and reading off cluster sizes.

---

## 2. System Architecture

```
 Optical tracking          Stream                 Formation             Delivery
 cameras (per stadium)     processing             engine                layer
 ┌──────────────────┐     ┌─────────────────┐    ┌──────────────────┐  ┌───────────────┐
 │ 16 fixed cameras  │     │ Azure Event Hubs │    │ Smoothing         │  │ WebSocket push │
 │ → x,y per player  │ ──▶ │ (ingest)         │ ─▶ │ Depth clustering  │─▶│ to broadcast   │
 │ ~25 frames/sec    │     │ Spark Structured │    │ Confidence scoring│  │ graphics + app │
 └──────────────────┘     │ Streaming (clean, │    │ Label smoothing   │  └───────────────┘
                           │ normalize, window)│    └──────────────────┘
                           └─────────────────┘              │
                                                             ▼
                                                   ┌──────────────────┐
                                                   │ Time-series store │
                                                   │ (Delta Lake) for   │
                                                   │ post-match analysis│
                                                   └──────────────────┘
```

**Component responsibilities**

| Component | Responsibility | Notes |
|---|---|---|
| Camera ingestion | Emit raw `{player_id, team, x, y, timestamp}` frames | ~25 Hz per player |
| Stream processing | Outlier rejection, coordinate normalization to 0–1 pitch space, rolling window average | Window length is the primary latency/stability trade-off |
| Formation engine | Depth clustering → formation string, confidence score, label smoothing | Stateless per tick; see §4 for algorithm |
| Delivery | Push formation + line markers over WebSocket | Target: sub-2s end-to-end for broadcast overlay use |
| Storage | Log every computed state to a time-series table | Feeds post-match "shape over time" analysis, not just live display |

---

## 3. Data Schema

### 3.1 Input: tracking frame (post-smoothing)

```json
{
  "match_id": "str",
  "timestamp": "ISO8601",
  "team": "A | B",
  "players": [
    { "player_id": "str", "x": 0.0, "y": 0.0, "role": "GK | OUTFIELD" }
  ]
}
```
- `x`: normalized depth, 0 = own goal line, 1 = opponent goal line.
- `y`: normalized width, 0–1 across the pitch.
- Goalkeeper is excluded from formation clustering.

### 3.2 Output: formation state

```json
{
  "match_id": "str",
  "timestamp": "ISO8601",
  "team": "A | B",
  "formation": "4-3-3",
  "line_counts": [4, 3, 3],
  "line_mean_depth": [0.12, 0.41, 0.73],
  "confidence": 0.0,
  "smoothed": true
}
```

---

## 4. Algorithm Specification

### 4.1 Depth clustering (gap-based)

1. Take the 10 outfield players' smoothed `x` (depth) values.
2. Sort ascending.
3. Compute the 9 consecutive gaps between sorted values.
4. Select up to 3 largest gaps as candidate line breaks.
5. Keep only candidate gaps ≥ `MIN_GAP` (tunable; prototype default `0.07`).
6. Split the sorted list at the kept gap positions → 1–4 clusters.
7. `line_counts` = cluster sizes, ordered defense → attack.
8. `formation` = `line_counts.join('-')`.

### 4.2 Confidence scoring

```
confidence = min(100, (smallest_kept_gap / MIN_GAP) * 55 + 30)
```
A simple proxy for how decisively the weakest accepted split clears the threshold.
**Not a substitute for a proper statistical test** — see §6, item 1.

### 4.3 Label smoothing

The formation *label* (not just the coordinates) is passed through an exponential
moving average / majority-vote over the last N ticks before being emitted, to avoid
flickering between adjacent readings (e.g. `4-3-3` ↔ `4-2-3-1`) during genuinely
transitional phases of play.

---

## 5. Tuning Parameters

| Parameter | Prototype default | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `MIN_GAP` | 0.07 | Fewer, more stable lines; may miss real shape changes (e.g. back 3 → back 5) | More lines detected; sensitive to noise from a single drifting player |
| Smoothing window | 20–45s | More stable label, slower to reflect real changes | Faster reaction, more flicker |
| Label smoothing depth (N ticks) | tunable | Suppresses transient mislabels | Reacts faster but noisier on-screen |

Recommendation: run two parallel configurations — a fast/low-window instance for
broadcast overlays, and a slow/high-window instance for the "formation for this phase
of play" statistic. These serve different audiences and should not share one config.

---

## 6. Known Limitations & Future Work

1. **Confidence score is heuristic, not statistical.** Replace with a proper measure
   (e.g. silhouette score across candidate cluster counts) before relying on it for
   any automated decision.
2. **Depth-only clustering ignores width.** A back four that's narrow (compact
   defensive block) vs. stretched wide reads identically in this model. A second
   clustering pass on the `y` axis within each line would recover this distinction.
3. **No template matching.** The engine reports raw cluster counts (e.g. `4-3-2-1`)
   rather than mapping to the nearest conventional formation name a viewer would
   recognize (e.g. `4-3-3` with an advanced 8). Template matching against a library of
   known shapes is a natural v2.
4. **Single-frame independence.** Each tick is computed independently aside from
   label smoothing. A proper implementation would track individual player identity
   across ticks to detect *role* changes (e.g. a fullback overlapping into midfield),
   not just aggregate counts.

---

## 7. Reference Prototype

A working browser-based prototype implementing §4.1–4.3 against simulated tracking
data is available as `formation_display_demo.html` (delivered separately). It
visualizes the live gap-clustering trace alongside the pitch and lets you trigger a
shape change (defensive block ↔ high press) to observe the label update.
