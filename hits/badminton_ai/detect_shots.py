"""Stage 5: shuttle trajectory -> stroke events.

Finds the frames where a racket hit the shuttle, from the stage-2 trajectory
alone. Two independent detectors are unioned, because each finds strokes the
other cannot:

1. **Acceleration peaks** catch sharp, fast strokes. "The shuttle reversed
   direction" is not enough -- it also reverses at the apex of every clear.
   What separates them is *abruptness*: gravity turns the shuttle over many
   frames, a racket turns it in one (apex ~3 px/frame^2, stroke ~32).

2. **Ballistic divergence** catches slow far-court strokes, where apparent
   motion is a few px/frame and a stroke's |dv| is buried in tracker jitter.
   Fit a ballistic arc to the preceding frames, extrapolate, and measure how
   far reality departs from it. The deviation accumulates over frames instead
   of living in one noisy sample.

Then rejection rules, each one added because of a measured failure:

* camera cuts / close-ups   -> split runs at cuts, drop runs that barely move
* frozen tracker            -> drop stretches pinned to one coordinate
* tracker excursions        -> drop 1-2 frame jumps whose neighbours share one arc
* ballistic echoes          -> drop a fit window that contains an acceleration peak
* serve release            -> a racket can only strike a shuttle already in flight
* no departure              -> a stroke sends the shuttle away; a tumble over the net does not
* floor landing           -> low in the frame AND never rises again
* run edges                 -> derivatives at the first/last frames are artefacts

And one recovery rule:

* off-screen strokes        -> a shuttle cannot come back faster than it left
                               unless something hit it out of sight

All frame-based constants assume the stated fps is the true content rate.
Check for padded frames at ingest (``trajectory_stats()['duplicate_fraction']``)
before trusting any output.

Rebuild note (Oct 2026): this module was re-implemented from the design notes
in ``docs/stage5-shot-detection.md`` after the original code was lost. The
constants are the documented ones; the exact ballistic score formula is a
reconstruction, and three details are new in the rebuild and still need
checking on real clips: the earliest-peak-of-a-cluster rule for ballistic
candidates, suppression of echoes just after a rejected release or landing,
and the 0.28 s merge gap.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from typing import Iterable, Optional

import numpy as np

from .types import DetectionResult, RejectedCandidate, ShotEvent, ShuttlePosition

Point = tuple[int, float, float]   # (frame, x, y)


@dataclass(frozen=True)
class Config:
    # --- trajectory cleaning ---------------------------------------------
    frozen_min_run: int = 4           # >= this many frames at one coordinate = tracker latched
    frozen_tol_px: float = 1.0        # a real apex holds still for at most ~3 frames
    excursion_max_len: int = 2        # tracker jumps to a shirt or racket and back
    excursion_context: int = 3        # clean frames needed on each side
    excursion_min_px: float = 8.0
    excursion_ratio: float = 3.0      # jump must exceed this x the context fit residual
    excursion_max_bend: float = 1.7   # frame heights / s^2: velocity change free flight allows
                                      # across a glitch; strokes measured 2.5-6x more
    max_interp_gap: int = 3           # bridge dropouts up to this many frames
    min_run_len: int = 10             # shorter runs cannot support the detectors
    min_run_median_step_px: float = 1.5   # runs that barely move: close-ups, static latch
    edge_margin: int = 3              # derivatives at run edges are artefacts

    # --- detector 1: acceleration peaks ------------------------------------
    accel_k: float = 6.0              # threshold = median + k * MAD of |dv| (MAD: strokes are outliers)
    accel_floor: float = 8.0          # px/frame^2; apex turnover is ~3

    # --- detector 2: ballistic divergence ----------------------------------
    ballistic_window: int = 8         # frames fitted before the candidate
    ballistic_horizon: int = 5        # frames extrapolated after it
    ballistic_threshold: float = 1.3  # deviation / distance travelled; noise <= ~0.9
    ballistic_min_speed: float = 2.0  # px/frame floor on the normaliser

    # --- rejection rules ---------------------------------------------------
    min_inflight_speed_fraction: float = 0.20   # release ~0.11, slowest real stroke ~0.34
    inflight_lookback: int = 8
    min_departure_speed_fraction: float = 0.25  # net tumbles, latches 0.11-0.25; slowest real stroke 0.57
    departure_frames: int = 4
    floor_y_fraction: float = 0.75    # highest real stroke 0.68, bounces 0.80-0.85
    floor_rise_fraction: float = 0.03 # bounces rose 0.000-0.007; only applied below floor_y_fraction
    floor_rise_window: int = 15
    floor_min_frames_after: int = 5   # too little trajectory to judge -> keep

    # --- off-screen inference ----------------------------------------------
    offscreen_speed_ratio: float = 2.0     # free flight 3.1-5.3, struck 0.89-1.29
    offscreen_min_gap: int = 4
    offscreen_max_gap_s: float = 3.0
    offscreen_return_top_fraction: float = 0.30
    offscreen_speed_frames: int = 3

    # --- merging -------------------------------------------------------------
    min_stroke_gap_s: float = 0.28        # < 8 frames at 25 fps: a release 8 frames before its
                                          # serve must survive merging (and is rejected separately)

    # --- ingest checks -------------------------------------------------------
    duplicate_warn_fraction: float = 0.15  # padded 30->60 fps clip measured 0.29

    def with_sensitivity(self, s: float) -> "Config":
        """s < 1 finds more strokes (and more false positives)."""
        return replace(self, accel_k=self.accel_k * s, accel_floor=self.accel_floor * s,
                       ballistic_threshold=self.ballistic_threshold * s)


DEFAULT = Config()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_ball_csv(path: str) -> list[ShuttlePosition]:
    """Read TrackNetV3's ``<video>_ball.csv`` (Frame, Visibility, X, Y)."""
    out: list[ShuttlePosition] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            vis = float(row.get("Visibility", 0) or 0) >= 0.5
            x, y = float(row.get("X", 0) or 0), float(row.get("Y", 0) or 0)
            if vis and x == 0 and y == 0:
                vis = False
            out.append(ShuttlePosition(int(float(row["Frame"])), vis,
                                       x if vis else None, y if vis else None))
    return out


def trajectory_stats(positions: list[ShuttlePosition], min_dropout: int = 5) -> dict:
    """Health check to run *before* touching any threshold.

    If a clip comes up short on strokes, the tracked fraction is the first
    thing to look at: no trajectory, nothing to analyse.
    """
    n = len(positions)
    vis = [p for p in positions if p.visible]
    dup = sum(1 for a, b in zip(vis, vis[1:])
              if b.frame_index == a.frame_index + 1 and a.x == b.x and a.y == b.y)
    dropouts, start = [], None
    for p in positions:
        if not p.visible and start is None:
            start = p.frame_index
        elif p.visible and start is not None:
            if p.frame_index - start >= min_dropout:
                dropouts.append((start, p.frame_index - 1))
            start = None
    if start is not None and positions and positions[-1].frame_index - start + 1 >= min_dropout:
        dropouts.append((start, positions[-1].frame_index))
    return {
        "frames": n,
        "tracked_fraction": len(vis) / n if n else 0.0,
        "duplicate_fraction": dup / max(1, len(vis) - 1),
        "dropouts": dropouts,
    }


# ---------------------------------------------------------------------------
# Trajectory cleaning
# ---------------------------------------------------------------------------

def _drop_frozen(visible: list[Point], min_run: int = 4, tol: float = 1.0) -> list[Point]:
    """Discard stretches where the reported position stops changing.

    A shuttle in flight is never stationary for long -- even at the apex of a
    clear it holds still for at most a couple of frames. When TrackNet reports
    the same coordinate for several frames running, it has latched onto
    something static. Observed mid-rally on broadcast footage: frames 234-272
    pinned at exactly (1275, 465) while play continued. The transitions into
    and out of such a stretch are enormous fake accelerations, so the points
    are dropped rather than kept.
    """
    if len(visible) < min_run:
        return visible
    keep = [True] * len(visible)
    i = 0
    while i < len(visible):
        j = i + 1
        while (j < len(visible)
               and visible[j][0] == visible[j - 1][0] + 1
               and abs(visible[j][1] - visible[i][1]) <= tol
               and abs(visible[j][2] - visible[i][2]) <= tol):
            j += 1
        if j - i >= min_run:
            for k in range(i, j):
                keep[k] = False
        i = j
    return [p for p, k in zip(visible, keep) if k]


def _drop_excursions(visible: list[Point], cfg: Config, fps: float, height: int) -> list[Point]:
    """Discard 1-2 frame jumps off the flight path.

    The frames either side of a tracker glitch move at the same velocity, give
    or take what gravity and drag can bend in that time. Either side of a
    stroke they do not -- and a parabola through both sides cuts the corner of
    the stroke, so the velocity test is what keeps strokes in.
    """
    k = cfg.excursion_context
    n = len(visible)
    if n < 2 * k + 1:
        return visible
    bend = cfg.excursion_max_bend * height / fps ** 2
    F = np.array([p[0] for p in visible], float)
    P = np.array([[p[1], p[2]] for p in visible], float)
    drop: set[int] = set()
    for j in range(k, n - k):
        for m in range(1, cfg.excursion_max_len + 1):
            if j + m + k > n:
                break
            ctx = list(range(j - k, j)) + list(range(j + m, j + m + k))
            if F[ctx[-1]] - F[ctx[0]] > 2 * k + m:
                continue
            v_before = (P[j - 1] - P[j - k]) / (F[j - 1] - F[j - k])
            v_after = (P[j + m + k - 1] - P[j + m]) / (F[j + m + k - 1] - F[j + m])
            if np.linalg.norm(v_after - v_before) > bend * (m + k):
                continue
            t = F[ctx]
            cx, cy = np.polyfit(t, P[ctx, 0], 2), np.polyfit(t, P[ctx, 1], 2)
            resid = float(np.max(np.hypot(np.polyval(cx, t) - P[ctx, 0], np.polyval(cy, t) - P[ctx, 1])))
            te = F[j:j + m]
            dev = float(np.min(np.hypot(np.polyval(cx, te) - P[j:j + m, 0], np.polyval(cy, te) - P[j:j + m, 1])))
            if dev >= max(cfg.excursion_min_px, cfg.excursion_ratio * resid):
                drop.update(range(j, j + m))
                break
    return [p for i, p in enumerate(visible) if i not in drop]


def _interpolate(visible: list[Point], max_gap: int, cuts: set[int]) -> list[Point]:
    """Linearly bridge short dropouts, never across a camera cut."""
    if len(visible) < 2:
        return list(visible)
    out: list[Point] = [visible[0]]
    for (f0, x0, y0), (f1, x1, y1) in zip(visible, visible[1:]):
        gap = f1 - f0
        if 1 < gap <= max_gap + 1 and not any(f0 < c <= f1 for c in cuts):
            for step in range(1, gap):
                t = step / gap
                out.append((f0 + step, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        out.append((f1, x1, y1))
    return out


@dataclass
class _Run:
    frames: np.ndarray      # (n,) consecutive frame indices
    P: np.ndarray           # (n, 2) positions

    @property
    def v(self) -> np.ndarray:          # v[i] = P[i+1] - P[i]
        return np.diff(self.P, axis=0)

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.v, axis=1)


def _split_runs(track: list[Point], cuts: set[int]) -> list[_Run]:
    """Split into runs of consecutive frames within one camera shot."""
    runs, cur = [], []
    for p in track:
        if cur and (p[0] != cur[-1][0] + 1 or p[0] in cuts):
            runs.append(cur)
            cur = []
        cur.append(p)
    if cur:
        runs.append(cur)
    return [_Run(np.array([p[0] for p in r]), np.array([[p[1], p[2]] for p in r], float))
            for r in runs]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _local_maxima(sig: np.ndarray, lo: int, hi: int, thr: float) -> list[int]:
    out = []
    for i in range(max(lo, 0), min(hi, len(sig))):
        if not np.isfinite(sig[i]) or sig[i] < thr:
            continue
        left = sig[i - 1] if i > 0 else -np.inf
        right = sig[i + 1] if i + 1 < len(sig) else -np.inf
        if sig[i] >= left and sig[i] > right:
            out.append(i)
    return out


def _accel_signal(run: _Run) -> np.ndarray:
    """|dv| at each frame (NaN at the ends). Raw, not smoothed: a 3-frame
    average collapsed a real stroke's |dv| from ~8 to 2.2."""
    a = np.full(len(run.frames), np.nan)
    v = run.v
    if len(v) >= 2:
        a[1:-1] = np.linalg.norm(np.diff(v, axis=0), axis=1)
    return a


def _ballistic_signal(run: _Run, cfg: Config, clip_speed: float = 0.0) -> np.ndarray:
    """Deviation from a fitted arc, as a fraction of distance actually travelled.

    score(i) fits frames i-W+1..i and compares the extrapolation with
    i+1..i+H, so a stroke at frame i scores highest at i.

    Normalised by *observed* speed. Normalising by the extrapolation's own
    predicted travel was tried and was wrong: a decelerating quadratic can
    predict the shuttle nearly stopping, the divisor collapses and the score
    explodes (one frame scored 3.5x too high that way).
    """
    W, H = cfg.ballistic_window, cfg.ballistic_horizon
    n = len(run.frames)
    s = np.full(n, np.nan)
    if n < W + H:
        return s
    t_fit = np.arange(W, dtype=float)
    t_pred = np.arange(W, W + H, dtype=float)
    k = np.arange(1, H + 1, dtype=float)
    speed = run.speed
    slow = speed < cfg.min_inflight_speed_fraction * clip_speed
    for i in range(W - 1, n - H):
        # The fit assumes free flight. A window that is mostly a shuttle held
        # in the hand is not an arc, and extrapolating it echoes the release.
        if slow[i - W + 1:i].mean() > 1 / 3:
            continue
        seg = run.P[i - W + 1:i + 1]
        cx = np.polyfit(t_fit, seg[:, 0], 2)
        cy = np.polyfit(t_fit, seg[:, 1], 2)
        pred = np.stack([np.polyval(cx, t_pred), np.polyval(cy, t_pred)], axis=1)
        dev = np.linalg.norm(run.P[i + 1:i + 1 + H] - pred, axis=1)
        v_obs = max(float(np.mean(speed[i:i + H])), cfg.ballistic_min_speed)
        s[i] = float(dev.sum() / (v_obs * k.sum()))
    return s


# ---------------------------------------------------------------------------
# Rejection rules
# ---------------------------------------------------------------------------

def _released_not_struck(run: _Run, i: int, clip_median_speed: float, cfg: Config) -> Optional[float]:
    """A racket can only strike a shuttle already in flight.

    A serve release (shuttle held still, then dropped) is a genuine abrupt
    velocity change and fires both detectors -- at confidence 1.00 on the
    clip that exposed it, against 0.10 for the real contact 8 frames later.
    That is why this runs *before* merging: suppression would have kept the
    release and thrown away the stroke.
    """
    sp = run.speed[max(0, i - cfg.inflight_lookback):i]
    if len(sp) < 3 or clip_median_speed <= 0:
        return None
    frac = float(np.mean(sp)) / clip_median_speed
    return frac if frac < cfg.min_inflight_speed_fraction else None


def _sent_nowhere(run: _Run, i: int, clip_median_speed: float, cfg: Config) -> Optional[float]:
    """A stroke sends the shuttle away; a tumble over the net or a latched tracker does not."""
    sp = run.speed[i:i + cfg.departure_frames]
    if len(sp) < cfg.departure_frames or clip_median_speed <= 0:
        return None
    frac = float(np.mean(sp)) / clip_median_speed
    return frac if frac < cfg.min_departure_speed_fraction else None


def _floor_landing(run: _Run, i: int, height: int, cfg: Config) -> Optional[float]:
    """Low in the picture AND never comes back up.

    The first version of this rule used post-event speed ("a landed shuttle
    stops"). It failed by a wide margin on a shuttle that landed and *skidded*
    at 0.91x the clip median speed. Height plus rise needs both conditions, so
    a genuine low retrieval -- near the floor but sent back up -- survives.
    Framing-dependent: a camera showing only the near half-court puts the
    floor higher in the picture. Court calibration is the fix for that.
    """
    y = run.P[i, 1]
    if y / height <= cfg.floor_y_fraction:
        return None
    after = run.P[i + 1:i + 1 + cfg.floor_rise_window, 1]
    if len(after) < cfg.floor_min_frames_after:
        return None
    rise = (y - float(after.min())) / height       # image y grows downward
    return rise if rise < cfg.floor_rise_fraction else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _features(run: _Run, i: int) -> dict:
    v = run.v
    pre = v[max(0, i - 3):i]
    post = v[i:i + 3]
    end = min(i + 10, len(run.P) - 1)
    return {
        "pre_velocity": tuple(float(c) for c in pre.mean(axis=0)) if len(pre) else None,
        "post_velocity": tuple(float(c) for c in post.mean(axis=0)) if len(post) else None,
        "post_travel_px": float(np.linalg.norm(run.P[end] - run.P[i])) if end > i else None,
    }


def detect_shots(
    positions: list[ShuttlePosition],
    fps: float,
    width: int,
    height: int,
    cut_frames: Optional[Iterable[int]] = None,
    sensitivity: float = 1.0,
    config: Config = DEFAULT,
) -> DetectionResult:
    """Detect strokes in a shuttle trajectory.

    Args:
        positions: stage 2 output for the clip, one entry per frame.
        fps: the true *content* frame rate (verify at ingest).
        width, height: frame size in pixels.
        cut_frames: frames where a new camera shot begins (``scene_cuts.py``).
        sensitivity: < 1.0 finds more strokes and more false positives.
    """
    cfg = config.with_sensitivity(sensitivity) if sensitivity != 1.0 else config
    cuts = set(cut_frames or [])
    rejected: list[RejectedCandidate] = []

    visible = sorted((p.frame_index, float(p.x), float(p.y))
                     for p in positions if p.visible and p.x is not None)
    cleaned = _drop_frozen(visible, cfg.frozen_min_run, cfg.frozen_tol_px)
    n_unfrozen = len(cleaned)
    cleaned = _drop_excursions(cleaned, cfg, fps, height)
    track = _interpolate(cleaned, cfg.max_interp_gap, cuts)

    runs, dropped_runs = [], 0
    for r in _split_runs(track, cuts):
        if len(r.frames) < cfg.min_run_len or float(np.median(r.speed)) < cfg.min_run_median_step_px:
            dropped_runs += 1
            continue
        runs.append(r)

    stats = trajectory_stats(positions)
    stats.update(runs=len(runs), dropped_runs=dropped_runs,
                 frozen_points_dropped=len(visible) - n_unfrozen,
                 excursion_points_dropped=n_unfrozen - len(cleaned))
    if stats["duplicate_fraction"] > cfg.duplicate_warn_fraction:
        stats["warning"] = (f"{stats['duplicate_fraction']:.0%} of consecutive positions are identical: "
                            "the video is probably padded (e.g. 30 fps content in a 60 fps container). "
                            "Decimate before detection; every frame-based constant assumes true fps.")
    if not runs:
        return DetectionResult([], rejected, stats)

    # Clip-adaptive thresholds.
    all_acc = np.concatenate([_accel_signal(r) for r in runs])
    all_acc = all_acc[np.isfinite(all_acc)]
    med = float(np.median(all_acc))
    mad = float(np.median(np.abs(all_acc - med)))
    accel_thr = max(cfg.accel_floor, med + cfg.accel_k * mad)
    clip_speed = float(np.median(np.concatenate([r.speed for r in runs])))
    stats.update(accel_threshold=accel_thr, clip_median_speed=clip_speed)

    # (frame, confidence, detector, run, index)
    cands: list[tuple[int, float, str, _Run, int]] = []
    fit_echoes: list[RejectedCandidate] = []
    for r in runs:
        n = len(r.frames)
        lo, hi = cfg.edge_margin, n - cfg.edge_margin
        acc = _accel_signal(r)
        peaks = _local_maxima(acc, lo, hi, accel_thr)
        for i in peaks:
            cands.append((int(r.frames[i]), min(1.0, acc[i] / accel_thr - 1.0), "accel", r, i))
        bal = _ballistic_signal(r, cfg, clip_speed)
        last = -10 ** 9
        for i in _local_maxima(bal, lo, hi, cfg.ballistic_threshold):
            # A window that straddles a stroke fits badly and echoes it a few
            # frames later -- on slow strokes the echo can outscore the true
            # frame. The change happened at the first peak of a cluster.
            if i - last <= cfg.ballistic_horizon:
                continue
            last = i
            echo = [a for a in peaks if i - cfg.ballistic_window < a < i]
            if echo:
                fit_echoes.append(RejectedCandidate(int(r.frames[i]), "ballistic",
                                                    "echo of an acceleration peak (frames after it)",
                                                    float(i - echo[-1])))
                continue
            cands.append((int(r.frames[i]), min(1.0, bal[i] / cfg.ballistic_threshold - 1.0),
                          "ballistic", r, i))

    kept = []
    for c in cands:
        f, conf, det, r, i = c
        frac = _released_not_struck(r, i, clip_speed, cfg)
        if frac is not None:
            rejected.append(RejectedCandidate(f, det, "released, not struck (speed before / clip median)", frac))
            continue
        rise = _floor_landing(r, i, height, cfg)
        if rise is not None:
            rejected.append(RejectedCandidate(f, det, "floor landing (rise after / frame height)", rise))
            continue
        dep = _sent_nowhere(r, i, clip_speed, cfg)
        if dep is not None:
            fit_echoes.append(RejectedCandidate(f, det, "sent nowhere (speed after / clip median)", dep))
            continue
        kept.append(c)

    # A rejected release or landing is one physical event; the detectors can
    # echo it a few frames later (e.g. a fit window straddling the release).
    gap_frames = max(1, int(round(cfg.min_stroke_gap_s * fps)))
    vetoed = [c.frame_index for c in rejected]
    echoes = [c for c in kept if any(0 < c[0] - v < gap_frames for v in vetoed)]
    for c in echoes:
        rejected.append(RejectedCandidate(c[0], c[2], "echo of a rejected event (frames after it)",
                                          float(min(c[0] - v for v in vetoed if c[0] > v))))
    echo_ids = {id(c) for c in echoes}
    kept = [c for c in kept if id(c) not in echo_ids]
    rejected.extend(fit_echoes)

    # Off-screen strokes: the shuttle left the top of the frame and came back.
    max_gap = cfg.offscreen_max_gap_s * fps
    k = cfg.offscreen_speed_frames
    for a, b in zip(runs, runs[1:]):
        gap = int(b.frames[0] - a.frames[-1] - 1)
        if not (cfg.offscreen_min_gap <= gap <= max_gap):
            continue
        if any(a.frames[-1] < c <= b.frames[0] for c in cuts):
            continue
        if len(a.v) < k or len(b.v) < k:
            continue
        exiting_up = float(np.mean(a.v[-k:, 1])) < 0
        returning_from_top = (b.P[0, 1] < cfg.offscreen_return_top_fraction * height
                              and float(np.mean(b.v[:k, 1])) > 0)
        if not (exiting_up and returning_from_top):
            continue
        ratio = float(np.mean(a.speed[-k:])) / max(float(np.mean(b.speed[:k])), 1e-6)
        f_last = int(a.frames[-1])
        if ratio < cfg.offscreen_speed_ratio:
            # Event at the last sighting: the far player's stroke is what drives
            # the shuttle out of the top of the frame, so the gap is the flight
            # *after* contact. Gap midpoint was 15-16 frames late; this is 1-2.
            conf = min(1.0, cfg.offscreen_speed_ratio / ratio - 1.0)
            kept.append((f_last, conf, "offscreen", a, len(a.frames) - 1))
        else:
            rejected.append(RejectedCandidate(f_last, "offscreen", "free flight (exit / return speed)", ratio))

    # Merge: one event per stroke, keep the highest-confidence frame.
    kept.sort(key=lambda c: -c[1])
    chosen: list[list] = []
    for c in kept:
        clash = next((ch for ch in chosen if abs(ch[0][0] - c[0]) < gap_frames), None)
        if clash is None:
            chosen.append([c, {c[2]}])
        else:
            clash[1].add(c[2])

    events = []
    for (f, conf, det, r, i), dets in sorted(chosen, key=lambda ch: ch[0][0]):
        inferred = det == "offscreen"
        feats = _features(r, i)
        if inferred:
            feats["post_velocity"] = feats["post_travel_px"] = None
        events.append(ShotEvent(
            frame_index=f, time_s=f / fps, confidence=round(float(conf), 3),
            detector="+".join(sorted(dets)),
            x=float(r.P[i, 0]), y=float(r.P[i, 1]), inferred=inferred, **feats))
    stats["events"] = len(events)
    return DetectionResult(events, rejected, stats)
