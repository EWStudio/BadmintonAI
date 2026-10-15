# Good-Badminton macOS Harness

![macOS](https://img.shields.io/badge/macOS-11%2B-lightgrey)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![Tests](https://img.shields.io/badge/tests-51%20passing-brightgreen)

macOS installers, compatibility patches and a hit-detection add-on for
[Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton), an open-source
badminton video analysis pipeline (player pose, shuttle detection, court
mapping, movement statistics, position heatmaps).

![Demo](docs/demo/demo.gif)

*The sample clip bundled with Good-Badminton: skeletons, player trails and
movement statistics from Good-Badminton; each detected hit named with the
player it is credited to. More in [Demo](#demo).*

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Testing](#testing)
- [Validation](#validation)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Dependency Notes](#dependency-notes)
- [Acknowledgements](#acknowledgements)

## Demo

Output for the 14-second sample clip bundled with Good-Badminton.

| File | Contents |
|---|---|
| [`detect_demo_hits.mp4`](docs/demo/detect_demo_hits.mp4) | Full annotated video |
| [`hits_verify.jpg`](docs/demo/hits_verify.jpg) | One tile per detected hit at the moment of contact |
| [`hits.csv`](docs/demo/hits.csv) | Every hit with time, player and evidence |
| [`match_heatmap.png`](docs/demo/match_heatmap.png), [`match_scatter.png`](docs/demo/match_scatter.png) | Where each player stood |

### Detected hits

![Verification sheet](docs/demo/hits_verify.jpg)

| # | Time | Player | Evidence | Detector |
|---|---|---|---|---|
| 1 | 1.57 s | near | distance 0.74 | accel |
| 2 | 3.03 s | far | distance 0.27 | accel |
| 3 | 3.47 s | near | distance 0.46 | accel+ballistic |
| 4 | 4.17 s | far | distance 0.77 | accel+ballistic |
| 5 | 4.87 s | near | alternation 0.65 | accel+ballistic |
| 6 | 5.67 s | far | distance 0.70 | accel+ballistic |

First 6 of 13; the full list is in [`hits.csv`](docs/demo/hits.csv).

### Player positions

<p align="center">
  <img src="docs/demo/match_heatmap.png" width="45%" alt="Position heatmap">
  <img src="docs/demo/match_scatter.png" width="45%" alt="Position scatter plot">
</p>

Against a hand-labelled answer key for the whole clip: 13 of 14 real hits
found (the low serve is missed), no false hits, and all 13 credited to the
right player. See [Validation](#validation).

## Features

- **One-command setup** for Apple Silicon and Intel Macs, isolated in its own virtual environment.
- **Upstream patches** for macOS: optional tkinter, Metal/CPU device selection, no export timeout, English output.
- **Hit detection and attribution**: when each shot was played and by which player, using the badminton-ai stage 5 detector and attribution.
- **False-hit filtering**: a hit counts only if a player's wrist was within reach of the shuttle, measured in metres through the court calibration; tracker glitches, detector echoes and a shuttle tumbling over the net are discarded.
- **Scoring against an answer key**: `score_hits.py` reports precision, recall and attribution accuracy on any hand-labelled clip.
- **Padded frame-rate detection**: finds duplicated frames (e.g. 25 fps content in a 30 fps file) and detects on the real frames.
- **TrackNetV3 input**: optionally replace Good-Badminton's shuttle track with a TrackNetV3 trajectory.
- **Verification outputs**: a contact sheet of every hit and an annotated video.

## Architecture

The system takes a badminton match video and produces two kinds of results:
how each player moved (court position, distance, speed, heatmaps) and when
each shot was hit and by which player. It has three layers: a macOS harness
that installs and patches the upstream project, the Good-Badminton analysis
pipeline, and a hit-detection stage that runs on the pipeline's output.

```mermaid
flowchart TB
    subgraph IN["Inputs"]
        direction LR
        V[/"Match video"/]
        TP[/"Court template frame"/]
    end

    subgraph MH["macOS harness"]
        direction LR
        SU["Setup scripts<br/>architecture detection · pinned dependencies"]
        PA["Patches<br/>device · tkinter · export · English"]
    end

    subgraph GB["Good-Badminton analysis"]
        direction LR
        R["Rally segmentation<br/>court-view frames"] --> P["Pose estimation<br/>YOLO11-pose · RTMPose"] & S["Shuttle detection<br/>YOLO11s"]
        P & S --> T["Player tracking<br/>position · distance · speed"]
        C["Court calibration<br/>pixels → metres"] --> T
    end

    J[("detections.jsonl")]
    O1["Movement results<br/>annotated video · heatmaps · stats"]
    TN[/"TrackNetV3 trajectory<br/>optional"/]

    subgraph HD["Hit detection"]
        direction LR
        A["Adapter<br/>frames · players · wrists"] --> I["Ingest check<br/>true frame rate"] --> D["Hit detector<br/>acceleration · ballistic arc"] --> RC["Reach check<br/>wrist within 1.3 m"] --> AT["Attribution<br/>wrist distance · alternation"]
    end

    O2["Hit results<br/>hits.csv · hits video · verification sheet"]

    IN --> GB
    MH -.-> GB
    GB --> J & O1
    J --> HD
    TN -.-> HD
    HD --> O2
```

| Layer | Component | Role | Technology |
|---|---|---|---|
| macOS harness | Setup scripts | Detect the CPU architecture, install pinned dependencies, download model weights | Bash, pip, lock file |
| macOS harness | Patches | Device selection, optional tkinter, no export timeout, English output | Python, reversible |
| Analysis | Rally segmentation | Keep frames that show the court | OpenCV template matching |
| Analysis | Pose estimation | Player skeletons and wrist positions | YOLO11-pose, RTMPose (ONNX Runtime) |
| Analysis | Shuttle detection | Shuttle position in each frame | YOLO11s (Ultralytics) |
| Analysis | Court calibration | Convert image pixels to court metres | OpenCV perspective transform |
| Analysis | Player tracking | Court position, distance and speed per player | NumPy |
| Hit detection | Adapter | Convert pipeline output: frame numbering, players, wrists, cuts | Python dataclasses |
| Hit detection | Ingest check | Detect padded video and recover the true frame rate | OpenCV frame differencing |
| Hit detection | Hit detector | Find strikes from abrupt acceleration and departures from a ballistic arc; drop tracker glitches, echoes and shuttles that are not sent anywhere | NumPy |
| Hit detection | Reach check | Keep a strike only if a wrist was within reach of the shuttle; merge the same player hitting twice within 0.5 s | NumPy, court homography |
| Hit detection | Attribution | Credit each strike to the player whose wrist was nearest in metres, alternating within a rally | NumPy |
| Hit detection | Renderer | Verification sheet and annotated video | OpenCV, FFmpeg |

**Stack:** Python · PyTorch · Ultralytics YOLO11 · RTMPose · ONNX Runtime · OpenCV · NumPy · SciPy · FFmpeg · pytest

### Design principles

- **Upstream kept intact.** Changes to Good-Badminton are scripted patches with backups and a one-command revert.
- **Stages exchange typed data.** Each stage reads and writes plain dataclasses and is tested on its own.
- **Reproducible environments.** Separate dependency sets per CPU architecture; Intel installs from a lock file of tested versions.
- **Results checked by hand.** Outputs are compared frame by frame against the video, and results, including failures, are recorded under [Validation](#validation).

## Requirements

| | Apple Silicon | Intel |
|---|---|---|
| macOS | 12+ | 11+ (12+ recommended) |
| Python | 3.12 (arm64) | 3.10–3.12 (x86_64) |
| PyTorch | 2.5.1 | 2.2.2 |
| Acceleration | Metal (MPS) | CPU only |
| Setup script | `setup_mac.sh` | `setup_mac_intel.sh` |

Both need an internet connection. Homebrew, Xcode Command Line Tools and a
system ffmpeg are optional. Each setup script refuses
to run on the other architecture.

## Installation

Clone the repository and run every command from its root.

### Apple Silicon

```bash
bash setup_mac.sh
```

### Intel

Install Python 3.12.10 first if no 3.10–3.12 is present (3.12.10 is the last
3.12 release with a macOS installer):

```bash
curl -LO https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg
sudo installer -pkg python-3.12.10-macos11.pkg -target /
```

Then:

```bash
bash setup_mac_intel.sh
```

Setup clones Good-Badminton into `vendor/`, builds the virtual environment
(`.venv` or `.venv-intel`), downloads the model weights (~91 MB), links an
ffmpeg if none is installed, and applies the patches.

### Updating

```bash
git pull
```

`vendor/`, the virtual environment and all outputs are git-ignored and left
untouched.

## Usage

Activate the environment in every new terminal, from the repository root
(`.venv` on Apple Silicon):

```bash
source .venv-intel/bin/activate
cd vendor/Good-Badminton
```

### 1. Analyze a video

```bash
python main.py --video-path videos/demo.mp4 --template-path templates/demo.png \
    --pose-family yolo-pose --display false
```

A window shows the auto-detected court corners: **Enter** accepts, **M**
switches to clicking the four corners manually (top-left, top-right,
bottom-right, bottom-left). Corners are cached in the output directory.

For your own footage, extract a template frame that shows the whole court:

```bash
ffmpeg -ss 1 -i match.mp4 -frames:v 1 templates/match.png
```

| Pose model | Flags |
|---|---|
| YOLO11n-pose (default) | `--pose-family yolo-pose` |
| RTMPose-s | `--pose-family rtmpose --pose-mode balanced` |
| RTMO-s | `--pose-family rtmo --pose-mode lightweight` |

Other `--pose-mode` values require rtmlib to download models at runtime.

### 2. Detect hits

```bash
python ../../scripts/detect_hits.py outputs/demo
python ../../scripts/detect_hits.py outputs/match --tracknet path/to/match_ball.csv
```

| Option | Description |
|---|---|
| `--tracknet BALL_CSV` | Use a TrackNetV3 trajectory for the shuttle; Good-Badminton supplies wrists only. Recommended. |
| `--sensitivity S` | Below 1.0 finds more hits and more false positives. Default 1.0. |
| `--reach METRES` | Drop hits with no wrist this close to the shuttle. Default 1.3; `0` turns the check off. |
| `--no-alternation` | Attribute by wrist distance only. |
| `--decimate {auto,off}` | Drop padded duplicate frames before detection. Default `auto`. |
| `--source VIDEO` | Source video for the padding check. Default: path in `metadata.json`. |
| `--video VIDEO` | Video to annotate. Default: the run's `detect_<name>.mp4`. |
| `--no-video`, `--no-montage` | Skip the annotated video or the contact sheet. |

### 3. Score against an answer key

Label the contact frames of a clip by hand (0-based, as in `hits.csv`):

```json
{"hits": [[47, "near"], [93, "far"], [104, "near"]]}
```

```bash
python ../../scripts/score_hits.py outputs/match match_answer_key.json
```

Prints hits found, false hits, misses and attribution accuracy, with the
frames of each error. A hit within 0.2 s of a labelled contact counts as found.

### 4. Inspect a run

```bash
python ../../scripts/compare_runs.py outputs/demo/detections.jsonl
python ../../scripts/compare_runs.py outputs/match/detections.jsonl --tracknet path/to/match_ball.csv
python ../../scripts/compare_runs.py outputs/match/detections.jsonl --frames 60-66
```

Prints frame coverage, player detection, shuttle gaps and, with `--tracknet`,
agreement with a TrackNetV3 trajectory. All frame numbers are 0-based.

### Web UI

```bash
pip install "gradio>=4.0.0"
python -m webui.app
```

## Output

Written to `vendor/Good-Badminton/outputs/<name>/`.

| File | Source | Contents |
|---|---|---|
| `detect_<name>.mp4` | Good-Badminton | Annotated video: skeletons, trails, mini-court, player stats |
| `detections.jsonl` | Good-Badminton | Per frame: player court position (m), speed, wrists, shuttle pixel position |
| `position_visualizations/` | Good-Badminton | Heatmaps and scatter plots per rally and per match |
| `metadata.json`, `court_annotations.txt` | Good-Badminton | Video info and cached court corners |
| `hits.json` | `detect_hits.py` | Events, rejected candidates, trajectory statistics, settings |
| `hits.csv` | `detect_hits.py` | One row per hit: frame, contact frame, time, player, attribution method and score |
| `hits_verify.jpg` | `detect_hits.py` | One tile per hit at contact |
| `detect_<name>_hits.mp4` | `detect_hits.py` | Annotated video with each hit named and a running tally |

Frame numbers in `hits.*` are 0-based. Player 0 is near (Good-Badminton
`lower`), player 1 is far (`upper`).

## Project Structure

```
.
├── setup_mac.sh               Apple Silicon setup
├── setup_mac_intel.sh         Intel setup
├── requirements-mac.txt
├── requirements-intel.txt
├── requirements-intel.lock    Exact versions tested on Intel
├── scripts/
│   ├── fetch_weights.sh       Model weights from the v0.1.0 release
│   ├── patch_for_mac.py       Upstream patches (--check, --revert)
│   ├── detect_hits.py         Hit detection CLI
│   ├── score_hits.py          Score hits.csv against an answer key
│   └── compare_runs.py        Run summary and TrackNetV3 comparison
├── hits/
│   ├── badminton_ai/          badminton-ai stage 5 and attribution
│   ├── adapter.py             Good-Badminton output → stage 5 inputs
│   ├── contact.py             Reach check and same-player merge
│   ├── pipeline.py            Detection → reach check → attribution
│   ├── score.py               Answer-key scoring
│   └── render.py              Contact sheet and annotated video
├── tests/
│   └── data/                  Tracking data and answer keys for both sample clips
├── docs/demo/                 Sample outputs shown in this README
└── vendor/Good-Badminton/     Created by setup, git-ignored
```

## Implementation Details

### Patches

`scripts/patch_for_mac.py` edits the cloned upstream code. It is idempotent,
writes `.orig` backups and is reverted with `--revert`.

| Patch | Reason |
|---|---|
| tkinter imported lazily | Homebrew Python ships without tkinter; upstream crashes on import, even for `--help`. |
| Device selection | Upstream only checks CUDA. Apple Silicon uses MPS; Intel uses CPU. Override with `GB_DEVICE=cpu\|mps\|cuda`. |
| No ffmpeg export timeout | A 180 s limit discards the final video on slow CPUs. |
| English output | Console output, `--help` and the web UI default to Chinese. |

### Hit detection

1. **Load** `detections.jsonl`: frames converted to 0-based, `lower`/`upper` mapped to near/far, wrists (COCO keypoints 9/10) assigned confidence 1.0, frames skipped by Good-Badminton treated as camera cuts.
2. **Ingest check**: frames that repeat their predecessor at a regular period are dropped and detection runs at the true frame rate.
3. **Detect**: badminton-ai stage 5 (acceleration peaks + ballistic divergence, with rejection rules for serve release, floor landing, frozen tracker and cuts, and off-screen inference), plus three rules added here:
   - **Tracker glitches**: 1–2 frame jumps are dropped when the frames either side move at the same velocity, give or take what gravity and drag allow. Either side of a real hit they do not.
   - **Ballistic echoes**: an arc fit whose window contains an acceleration peak is a repeat of that peak, not a new hit.
   - **No departure**: a hit sends the shuttle away; a candidate after which the shuttle moves at under 25% of the clip's median speed is a tumble over the net or a stuck detector.
4. **Reach check**: for each hit, the closest either player's wrist came to the shuttle from 4 frames before to 2 frames after, converted to metres at that player's position on court. No wrist within 1.3 m: rejected. For off-screen strokes, a wrist must come within 1.3 m of the top edge. The same player twice within 0.5 s is one hit.
5. **Attribute**: the reach distances become the evidence (1 at the shuttle, 0 at 1.3 m), constrained to alternate between players within a rally.
6. **Map back** to original frame numbers and render.

## Testing

```bash
pip install pytest
python -m pytest
```

51 tests: 32 for stage 5 and attribution, 8 for the adapter (frame
numbering, player mapping, cuts, padded-frame detection, decimation
round-trip, TrackNetV3 substitution), 6 for the reach check and 5 regression
tests that replay Good-Badminton's tracking of both sample clips against
their answer keys.

## Validation

### Installation

| Platform | Result |
|---|---|
| Intel, 2020 MacBook Air (i7-1060NG7, macOS 15.7.4, Python 3.12.10) | Full demo run; 426/426 frames analysed, shuttle in 407 (96%); 303 s for the 14.2 s clip (21× real time); hit detection identical to the Linux run |
| x86_64 Linux, Python 3.12 | Full demo run with both pose families; setup script in a simulated macOS environment |
| Apple Silicon | Dependency resolution only |

### Hit detection

Both sample clips distributed with Good-Badminton, every contact labelled by
hand ([`tests/data/`](tests/data)). Both are 25 fps content in a 30 fps file;
the ingest check removes the duplicated frames. Shuttle track from
Good-Badminton.

| Clip | Version | Reported | Found | False | Missed | Right player |
|---|---|---|---|---|---|---|
| Men's singles, 14 s, 14 hits | before | 17 | 13 | 4 | 1 | 7 of 13 |
| | after | 13 | 13 | **0** | 1 | **13 of 13** |
| Women's singles, 35 s, 28 hits | before | 42 | 27 | 15 | 1 | 16 of 27 |
| | after | 27 | 27 | **0** | 1 | **27 of 27** |

The miss in both clips is the serve. The false hits came from five sources:

| Source | Rule that removes it |
|---|---|
| Ball detector jumps to a player's white clothing for 1–2 frames | Tracker glitches |
| Arc fit repeats a real hit 4–7 frames later | Ballistic echoes |
| Off-screen inference fires on a high lift that simply fell back | Reach check (no wrist near the top edge) |
| Shuttle slowing high in the air after a clear | Ballistic echoes, reach check |
| Shuttle tumbling over the net, or the detector stuck on a wrist | No departure, same player within 0.5 s |

Each false hit also flipped every later attribution through the alternation
rule. Attribution additionally moved from pixel distance at 3 frames before
the detected frame (the near player scored 0 at most of their own hits) to
distance in metres over a window around contact.

The rules were designed on the men's singles clip. The women's singles clip
was then run unseen: false hits fell from 15 to 1. That remaining one, a
shuttle tumbling over the net, led to the no-departure rule, so the second
clip is not a fully independent test. Score your own footage with
[`score_hits.py`](#3-score-against-an-answer-key) before relying on the numbers.

## Known Limitations

- Singles only.
- Low serves are not detected.
- Thresholds were set on two broadcast clips from a high camera behind the baseline; other camera positions may need `--reach` adjusted.
- Camera cuts are only those Good-Badminton skips as non-court frames.
- Good-Badminton writes wrists without confidence, so spurious wrists count as evidence.
- Good-Badminton identifies players by court half (`upper`/`lower`); identity does not survive an end change.
- Good-Badminton's rally number is drawn on the video but not written to `detections.jsonl`.
- Court mapping uses four corners with no outlier rejection and assumes a 6.1 × 13.4 m doubles court.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No usable Python found` | Install Python 3.12.10 (see [Intel](#intel)). |
| `zsh: command not found: python` | Activate the virtual environment. |
| `Error checking audio track: … 'ffprobe'` | Harmless; the video is still exported. |
| Court window does not appear | Check behind other windows, or use the [web UI](#web-ui). |
| `Ingest check skipped: source video not found` | Run from `vendor/Good-Badminton` or pass `--source`. |
| `… consecutive positions are identical` warning | The video is padded and the ingest check did not run; pass `--source`. |
| Pose model download stalls | Use a pose family/mode pair listed in [Usage](#1-analyze-a-video). |
| Patch reports `expected code block not found` | Upstream changed; review the named file manually. |

## Dependency Notes

- Upstream pins `torch==2.5.1+cpu`; the `+cpu` build does not exist for macOS. The Apple Silicon requirements use `torch==2.5.1`.
- PyTorch 2.2.2 is the last release with Intel macOS wheels; torchvision is pinned to the matching 0.17.2.
- PyTorch 2.2.2 requires NumPy 1.x.
- OpenCV 4.10's Intel wheel requires macOS 12; on macOS 11 the setup script substitutes 4.9.0.80.
- `requirements-intel.lock` pins every package to the versions that were tested and is applied with `pip install -c`.
- ffmpeg is linked from `imageio-ffmpeg` (static build with libx264, macOS 11+).

## Acknowledgements

| Component | License |
|---|---|
| [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) | Apache-2.0 |
| [RTMPose / rtmlib](https://github.com/Tau-J/rtmlib) | Apache-2.0 |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| [TrackNetV3](https://github.com/qaz812345/TrackNetV3) | MIT |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) bundled ffmpeg | GPL |

The demo footage is the sample clip distributed with Good-Badminton.

`hits/badminton_ai/` is inspired by the stage 5 shot detection and attribution
in badminton-ai, and `tests/synth.py`, `tests/test_detect_shots.py` and
`tests/test_attribute.py` by its test suite.
