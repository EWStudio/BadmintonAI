"""Typed contracts between pipeline stages.

Every stage consumes and produces these dataclasses, so each stage can be run
and tested in isolation and swapped without touching its neighbours. Stages
agree on ``frame_index`` only; they do not share a frame representation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# COCO-17 keypoint indices used across the pipeline (YOLO-pose order).
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    decodable_frames: int          # counted by reading until EOF, never from the header

    @property
    def diagonal(self) -> float:
        return (self.width ** 2 + self.height ** 2) ** 0.5


@dataclass
class ShuttlePosition:
    """Stage 2 output, one per frame.

    ``visible`` is separate from the coordinates on purpose: "the model looked
    and saw nothing" is a different fact from "no data", and the difference is
    what you need when debugging tracking gaps.
    """
    frame_index: int
    visible: bool
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class PlayerPose:
    """Stage 3 output: one player in one frame. Keypoints are COCO-17, (x, y, conf)."""
    frame_index: int
    player_id: int                 # 0 = near court, 1 = far court (singles)
    bbox: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]]


@dataclass
class CourtTransform:
    """Stage 4 output. ``homography`` maps image pixels -> court metres."""
    homography: list[list[float]]
    inverse: list[list[float]]
    n_points: int
    n_inliers: int
    reprojection_error_m: float
    line_support: Optional[float] = None


@dataclass
class ShotEvent:
    """Stage 5 output: one stroke.

    Carries the trajectory features stage 6 needs (pre/post velocity, travel),
    because they are free at detection time and expensive to recover later.
    """
    frame_index: int
    time_s: float
    confidence: float
    detector: str                  # "accel", "ballistic", "accel+ballistic", "offscreen"
    x: Optional[float] = None
    y: Optional[float] = None
    inferred: bool = False         # True when the stroke happened while the shuttle was unseen
    pre_velocity: Optional[tuple[float, float]] = None    # px/frame, mean of 3 frames before
    post_velocity: Optional[tuple[float, float]] = None   # px/frame, mean of 3 frames after
    post_travel_px: Optional[float] = None                # distance covered in the next 10 frames
    player_id: Optional[int] = None
    attribution_score: Optional[float] = None
    attribution_method: Optional[str] = None              # "distance", "alternation", "elimination"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RejectedCandidate:
    frame_index: int
    detector: str
    reason: str
    value: float


@dataclass
class DetectionResult:
    events: list[ShotEvent]
    rejected: list[RejectedCandidate] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
