from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np

from .badminton_ai.types import KP_L_WRIST, KP_R_WRIST, ShuttlePosition

NEAR, FAR = 0, 1
REGION_TO_PLAYER = {"lower": NEAR, "upper": FAR}
PLAYER_NAMES = {NEAR: "near", FAR: "far", None: "unattributed"}
PLAYER_REGIONS = {NEAR: "lower", FAR: "upper", None: None}


@dataclass
class GBRun:
    run_dir: str
    name: str
    fps: float
    width: int
    height: int
    n_frames: int
    records: dict[int, dict]
    positions: list[ShuttlePosition]
    poses: dict[int, dict[int, np.ndarray]]
    cut_frames: list[int] = field(default_factory=list)
    source_video: str | None = None

    @property
    def diagonal(self) -> float:
        return (self.width ** 2 + self.height ** 2) ** 0.5


def _point(p):
    if not p or len(p) < 2:
        return None
    x, y = float(p[0]), float(p[1])
    return None if (x == 0 and y == 0) else (x, y)


def wrist_keypoints(hands: dict | None) -> np.ndarray | None:
    hands = hands or {}
    left, right = _point(hands.get("left")), _point(hands.get("right"))
    if left is None and right is None:
        return None
    k = np.zeros((17, 3))
    for idx, pt in ((KP_L_WRIST, left), (KP_R_WRIST, right)):
        if pt is not None:
            k[idx] = (pt[0], pt[1], 1.0)
    return k


def load_run(path: str) -> GBRun:
    jsonl = path if path.endswith(".jsonl") else os.path.join(path, "detections.jsonl")
    run_dir = os.path.dirname(os.path.abspath(jsonl))
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as fh:
        video = json.load(fh)["video"]

    records: dict[int, dict] = {}
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[int(rec["frame"]) - 1] = rec

    n = max(int(video["total_frames"]), (max(records) + 1) if records else 0)
    positions, poses = [], {}
    for f in range(n):
        rec = records.get(f)
        shuttle = _point(((rec or {}).get("shuttlecock") or {}).get("image"))
        positions.append(ShuttlePosition(f, shuttle is not None,
                                         shuttle[0] if shuttle else None,
                                         shuttle[1] if shuttle else None))
        if rec is None:
            continue
        for region, pid in REGION_TO_PLAYER.items():
            kps = wrist_keypoints(((rec.get("players") or {}).get(region) or {}).get("hands"))
            if kps is not None:
                poses.setdefault(f, {})[pid] = kps

    frames = sorted(records)
    cuts = [f for prev, f in zip(frames, frames[1:]) if f != prev + 1]

    return GBRun(run_dir=run_dir, name=video.get("name") or os.path.basename(run_dir),
                 fps=float(video["fps"]), width=int(video["width"]), height=int(video["height"]),
                 n_frames=n, records=records, positions=positions, poses=poses, cut_frames=cuts,
                 source_video=video.get("path"))


def with_tracknet(run: GBRun, ball_csv: str) -> tuple[GBRun, str | None]:
    from .badminton_ai.detect_shots import load_ball_csv
    by_frame = {p.frame_index: p for p in load_ball_csv(ball_csv)}
    warning = None
    if by_frame and abs((max(by_frame) + 1) - run.n_frames) > 2:
        warning = (f"TrackNetV3 CSV covers {max(by_frame) + 1} frames but the video has {run.n_frames}; "
                   "is it from the same clip?")
    positions = [by_frame.get(f, ShuttlePosition(f, False)) for f in range(run.n_frames)]
    return replace(run, positions=positions), warning


def find_duplicate_frames(video_path: str, max_width: int = 320,
                          changed_thresh: int = 12, rel: float = 0.1) -> list[int]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    changed, prev = [], None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if w > max_width:
            img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        changed.append(None if prev is None else int((np.abs(g - prev) > changed_thresh).sum()))
        prev = g
    cap.release()
    c = np.array([x if x is not None else np.nan for x in changed], float)
    dups = []
    for f in range(1, len(c)):
        local = np.nanmedian(c[max(1, f - 3):f + 4])
        if np.isfinite(local) and local > 0 and c[f] < rel * local:
            dups.append(f)
    return dups


def padding_period(dups: list[int]) -> int | None:
    if len(dups) < 4:
        return None
    gaps = np.diff(dups)
    base = int(np.min(gaps))
    if base < 2:
        return None
    return base if np.mean(gaps % base == 0) >= 0.9 else None


@dataclass
class Decimated:
    run: GBRun
    kept: list[int]
    dropped: list[int]
    true_fps: float

    def to_original(self, f: int) -> int:
        return self.kept[min(max(f, 0), len(self.kept) - 1)]


def decimate(run: GBRun, dup_frames: list[int]) -> Decimated:
    drop = set(dup_frames)
    kept = [f for f in range(run.n_frames) if f not in drop]
    new_index = {f: i for i, f in enumerate(kept)}
    positions = [ShuttlePosition(i, run.positions[f].visible, run.positions[f].x, run.positions[f].y)
                 for i, f in enumerate(kept)]
    poses = {new_index[f]: p for f, p in run.poses.items() if f in new_index}
    records = {new_index[f]: r for f, r in run.records.items() if f in new_index}
    cuts = sorted({next((i for i, k in enumerate(kept) if k >= c), len(kept) - 1) for c in run.cut_frames})
    true_fps = run.fps * len(kept) / run.n_frames
    sub = GBRun(run_dir=run.run_dir, name=run.name, fps=true_fps, width=run.width, height=run.height,
                n_frames=len(kept), records=records, positions=positions, poses=poses, cut_frames=cuts,
                source_video=run.source_video)
    return Decimated(sub, kept, sorted(drop), true_fps)
