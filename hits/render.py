from __future__ import annotations

import math
import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np

from .adapter import FAR, NEAR, PLAYER_NAMES

COLORS = {NEAR: (255, 0, 255), FAR: (0, 255, 255), None: (200, 200, 200)}
RING = (255, 255, 0)


def shuttle_xy(event, contact, positions) -> tuple[float, float]:
    for g in range(contact, event.frame_index + 1):
        p = positions[g] if g < len(positions) else None
        if p is not None and p.visible:
            return p.x, p.y
    return event.x, event.y


def label(event) -> str:
    if event.player_id is None:
        return "HIT (player unknown)"
    return f"HIT  {PLAYER_NAMES[event.player_id].upper()} player"


def _text(img, txt, org, color, scale=0.8, thick=2, bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + base + 4), bg, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class FrameSource:
    def __init__(self, path: str, order: list[int]):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.index = {f: k for k, f in enumerate(order)}

    def get(self, f: int) -> Optional[np.ndarray]:
        k = self.index.get(f)
        if k is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = self.cap.read()
        return img if ok else None

    def close(self):
        self.cap.release()


def write_montage(video: str, order: list[int], events, contacts, positions, out_path: str,
                  tile=(360, 270), cols=4) -> int:
    if not events:
        return 0
    src = FrameSource(video, order)
    tiles = []
    for n, (e, f) in enumerate(zip(events, contacts), 1):
        img = src.get(f)
        if img is None:
            img = src.get(e.frame_index)
        if img is None:
            img = np.zeros((tile[1] * 2, tile[0] * 2, 3), np.uint8)
        H, W = img.shape[:2]
        sx, sy = shuttle_xy(e, f, positions)
        ch = int(H * 0.55)
        cw = int(ch * tile[0] / tile[1])
        x0 = int(min(max(sx - cw / 2, 0), max(W - cw, 0)))
        y0 = int(min(max(sy - ch / 2, 0), max(H - ch, 0)))
        crop = img[y0:y0 + ch, x0:x0 + cw].copy()
        cv2.circle(crop, (int(sx - x0), int(sy - y0)), 14, RING, 2, cv2.LINE_AA)
        crop = cv2.resize(crop, tile, interpolation=cv2.INTER_AREA)
        color = COLORS[e.player_id]
        cv2.rectangle(crop, (0, 0), (tile[0] - 1, tile[1] - 1), color, 3)
        how = f"{e.attribution_method} {e.attribution_score:.2f}" if e.player_id is not None else "no wrist in range"
        _text(crop, f"#{n} f{e.frame_index} {e.time_s:.2f}s", (10, 26), (255, 255, 255), 0.6, 1)
        _text(crop, PLAYER_NAMES[e.player_id].upper(), (10, 54), color, 0.7, 2)
        _text(crop, how, (10, tile[1] - 34), (255, 255, 255), 0.5, 1)
        _text(crop, "off-screen inference" if e.inferred else e.detector, (10, tile[1] - 10),
              (255, 255, 255), 0.5, 1)
        tiles.append(crop)
    src.close()
    rows = math.ceil(len(tiles) / cols)
    tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
    sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(events)


def write_video(video: str, order: list[int], events, contacts, positions, fps: float, out_path: str,
                hold_s: float = 0.7) -> str:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hold = max(1, int(round(hold_s * fps)))
    marks = [(c, e, shuttle_xy(e, c, positions)) for e, c in zip(events, contacts)]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
             "-i", video, "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "copy", "-shortest", "-movflags", "+faststart", out_path],
            stdin=subprocess.PIPE)
        sink = proc.stdin
    else:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    counts = {NEAR: 0, FAR: 0, None: 0}
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f = order[k] if k < len(order) else None
        k += 1
        if f is not None:
            for cf, e, (sx, sy) in marks:
                if cf == f:
                    counts[e.player_id] += 1
                if cf <= f < cf + hold:
                    color = COLORS[e.player_id]
                    cv2.circle(img, (int(sx), int(sy)), 22, color, 3, cv2.LINE_AA)
                    txt = label(e)
                    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                    _text(img, txt, ((W - tw) // 2, 60), color, 1.1, 3)
        tally = f"Hits  near {counts[NEAR]}  |  far {counts[FAR]}" + \
                (f"  |  unknown {counts[None]}" if counts[None] else "")
        (tw, _), _ = cv2.getTextSize(tally, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        _text(img, tally, ((W - tw) // 2, H - 24), (255, 255, 255), 0.8, 2)
        if ffmpeg:
            sink.write(img.tobytes())
        else:
            writer.write(img)
    cap.release()
    if ffmpeg:
        sink.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing the hits video")
    else:
        writer.release()
    return out_path
