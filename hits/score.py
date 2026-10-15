from __future__ import annotations

import json
from dataclasses import dataclass, field

TOLERANCE_S = 0.2


@dataclass
class Score:
    reported: int
    true: int
    found: int
    right_player: int
    false: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)
    wrong_player: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.found / self.reported if self.reported else 1.0

    @property
    def recall(self) -> float:
        return self.found / self.true if self.true else 1.0

    def summary(self) -> str:
        return (f"{self.found} of {self.true} hits found, {len(self.false)} false, {len(self.missed)} missed | "
                f"precision {self.precision:.0%}, recall {self.recall:.0%} | "
                f"right player {self.right_player} of {self.found}")


def load_answer_key(path: str) -> list[tuple[int, str]]:
    with open(path, encoding="utf-8") as fh:
        return [(int(f), str(p)) for f, p in json.load(fh)["hits"]]


def score(hits: list[tuple[int, int, str]], key: list[tuple[int, str]], fps: float,
          tolerance_s: float = TOLERANCE_S) -> Score:
    """``hits`` are (frame, contact_frame, player); a hit matches the nearest unused true hit."""
    tol = tolerance_s * fps
    used: set[int] = set()
    s = Score(reported=len(hits), true=len(key), found=0, right_player=0)
    for frame, contact, player in hits:
        best = None
        for k, (f, _) in enumerate(key):
            d = min(abs(frame - f), abs(contact - f))
            if k not in used and d <= tol and (best is None or d < best[1]):
                best = (k, d)
        if best is None:
            s.false.append(frame)
            continue
        used.add(best[0])
        s.found += 1
        if key[best[0]][1] == player:
            s.right_player += 1
        else:
            s.wrong_player.append(frame)
    s.missed = [f for k, (f, _) in enumerate(key) if k not in used]
    return s
