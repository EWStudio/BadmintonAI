from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .adapter import Decimated, GBRun, decimate, padding_period
from .badminton_ai.attribute import LOOKBACK, attribute_strokes
from .badminton_ai.detect_shots import detect_shots
from .badminton_ai.types import RejectedCandidate, ShotEvent
from .contact import REACH_M, confirm_contacts

MIN_PADDING_SHARE = 0.05


@dataclass
class Hits:
    events: list[ShotEvent]
    contacts: list[int]
    rejected: list[RejectedCandidate]
    stats: dict
    note: Optional[str] = None


def padded(run: GBRun, dups: list[int]) -> Optional[Decimated]:
    if padding_period(dups) and len(dups) / max(1, run.n_frames) >= MIN_PADDING_SHARE:
        return decimate(run, dups)
    return None


def find_hits(run: GBRun, dec: Optional[Decimated] = None, sensitivity: float = 1.0,
              reach_m: float = REACH_M, alternation: bool = True) -> Hits:
    work = dec.run if dec else run
    to_orig = dec.to_original if dec else (lambda f: f)

    res = detect_shots(work.positions, work.fps, work.width, work.height,
                       cut_frames=work.cut_frames, sensitivity=sensitivity)
    checked = confirm_contacts(res.events, work, reach_m)
    attributed = attribute_strokes(checked.events, work.poses, work.positions, work.diagonal, work.fps,
                                   cut_frames=work.cut_frames, alternation=alternation, scores=checked.scores)

    events, contacts = [], []
    for e in attributed:
        contact = e.frame_index if e.inferred else max(0, e.frame_index - LOOKBACK)
        f = to_orig(e.frame_index)
        events.append(replace(e, frame_index=f, time_s=f / run.fps))
        contacts.append(to_orig(contact))
    rejected = sorted((replace(c, frame_index=to_orig(c.frame_index)) for c in res.rejected + checked.rejected),
                      key=lambda c: c.frame_index)
    stats = dict(res.stats)
    stats["dropouts"] = [(to_orig(x), to_orig(y)) for x, y in stats.get("dropouts", [])]
    stats["fps"] = work.fps
    if dec:
        stats["decimated"] = {"dropped_frames": dec.dropped, "true_fps": dec.true_fps}
    return Hits(events, contacts, rejected, stats, checked.note)
