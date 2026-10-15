#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/yo-WASSUP/Good-Badminton"
VENV=".venv-intel"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

say "1/7  Checking this Mac"

[[ "$(uname -s)" == "Darwin" ]] || die "This script is for macOS."
if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
  die "This is an Apple Silicon Mac (this shell may be running under Rosetta).
Use ./setup_mac.sh instead — it gets torch 2.5.1 and the Metal GPU."
fi
[[ "$(uname -m)" == "x86_64" ]] || die "Unexpected architecture: $(uname -m)"

MACOS="$(sw_vers -productVersion)"
MACOS_MAJOR="${MACOS%%.*}"
if (( MACOS_MAJOR >= 12 )); then
  OPENCV="4.10.0.84"
elif (( MACOS_MAJOR == 11 )); then
  OPENCV="4.9.0.80"
else
  die "macOS $MACOS is too old. onnxruntime 1.18.1 and the bundled ffmpeg both
need macOS 11 Big Sur or newer."
fi
CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo "  macOS $MACOS, $CPU, $(sysctl -n hw.ncpu) logical cores"
echo "  opencv pin for this macOS: $OPENCV"

say "2/7  Locating Python 3.12 / 3.11 / 3.10 (x86_64)"

PY=""
for ver in 3.12 3.11 3.10; do
  for candidate in \
      "/Library/Frameworks/Python.framework/Versions/$ver/bin/python$ver" \
      "/usr/local/bin/python$ver" \
      "/usr/local/opt/python@$ver/bin/python$ver" \
      "$(command -v "python$ver" || true)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    arch="$("$candidate" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")"
    if [[ "$arch" == "x86_64" ]]; then PY="$candidate"; break 2; fi
    echo "  skip $candidate (reports $arch)"
  done
done

if [[ -z "$PY" ]]; then
  cat >&2 <<'EOS'

No usable Python found. Need 3.10, 3.11 or 3.12 — NOT 3.13 or newer:
torch 2.2.2 (the last torch for Intel Macs) has no wheels past 3.12.

Easiest: the python.org installer. It is a prebuilt universal binary,
includes tkinter, and works on macOS 11+. 3.12.10 is the LAST 3.12 with a
Mac installer (later 3.12.x releases are source-only):
    curl -LO https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg
    sudo installer -pkg python-3.12.10-macos11.pkg -target /

(Homebrew also works on recent macOS: brew install python@3.12. On macOS
11/12 Homebrew is unsupported and may compile Python from source for a
long time — use the python.org installer instead.)

Then re-run this script.
EOS
  exit 1
fi
echo "  using $PY ($("$PY" --version 2>&1))"
"$PY" -c 'import tkinter' 2>/dev/null \
  && echo "  tkinter: present (file picker works)" \
  || echo "  tkinter: absent — fine, the patch makes it optional; always pass --template-path"

say "3/7  Fetching upstream"
if [[ -d vendor/Good-Badminton/badminton_analysis ]]; then
  echo "  already present; leaving it alone"
else
  mkdir -p vendor
  if xcode-select -p >/dev/null 2>&1 && git --version >/dev/null 2>&1; then
    git clone --depth 1 "$REPO.git" vendor/Good-Badminton
  else
    echo "  no Command Line Tools; downloading a tarball instead of git clone"
    tmp="$(mktemp -d)"
    curl -fsSL "$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$tmp"
    mv "$tmp"/Good-Badminton-main vendor/Good-Badminton
    rmdir "$tmp"
  fi
fi

say "4/7  Building $VENV"
if [[ -x "$VENV/bin/python" ]]; then
  have="$("$VENV/bin/python" -c 'import platform,sys; print(platform.machine(), "%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "broken")"
  case "$have" in
    "x86_64 3.10"|"x86_64 3.11"|"x86_64 3.12") echo "  reusing $VENV ($have)";;
    *) die "$VENV exists but is '$have' (copied from another Mac?). Delete it: rm -rf $VENV";;
  esac
else
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

REQ="requirements-intel.txt"; LOCK="requirements-intel.lock"
if [[ "$OPENCV" != "4.10.0.84" ]]; then
  tmpreq="$(mktemp -d)"
  sed "s/==4\.10\.0\.84/==$OPENCV/" "$REQ"  > "$tmpreq/req.txt"
  sed "s/==4\.10\.0\.84/==$OPENCV/" "$LOCK" > "$tmpreq/lock.txt"
  REQ="$tmpreq/req.txt"; LOCK="$tmpreq/lock.txt"
fi
echo "  installing (torch is ~150 MB; a few minutes)"
pip install -r "$REQ" -c "$LOCK"

python - <<'EOS'
import platform, sys, torch, numpy, cv2, onnxruntime as ort, ultralytics
print(f"\n  python      {platform.python_version()} / {platform.machine()}")
print(f"  torch       {torch.__version__}   (MPS used: no — CPU by default on Intel)")
print(f"  numpy       {numpy.__version__}")
print(f"  opencv      {cv2.__version__}")
print(f"  onnxruntime {ort.__version__}")
print(f"  ultralytics {ultralytics.__version__}")
if platform.machine() != "x86_64":
    sys.exit("\nSTOP: this venv is not x86_64.")
if not numpy.__version__.startswith("1."):
    sys.exit("\nSTOP: numpy 2 installed; torch 2.2.2 needs numpy 1.x.")
EOS

say "5/7  ffmpeg"
if [[ -e "$VENV/bin/ffmpeg" ]]; then
  echo "  using $VENV/bin/ffmpeg"
elif command -v ffmpeg >/dev/null 2>&1; then
  echo "  using system $(command -v ffmpeg)"
else
  ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$VENV/bin/ffmpeg"
  echo "  linked bundled ffmpeg -> $VENV/bin/ffmpeg (active whenever the venv is)"
fi
ffmpeg -hide_banner -version | head -1 | sed 's/^/  /'

say "6/7  Downloading weights (~91 MB)"
bash scripts/fetch_weights.sh vendor/Good-Badminton/weights

say "7/7  Patching for macOS"
python scripts/patch_for_mac.py --root vendor/Good-Badminton

cat <<EOS

================================================================
Setup complete. Run the bundled 14-second demo first:

    source $VENV/bin/activate
    cd vendor/Good-Badminton
    python main.py \\
        --video-path videos/demo.mp4 \\
        --template-path templates/demo.png \\
        --pose-family yolo-pose \\
        --display false

A window shows the auto-detected court corners.
  Enter / Y  accept
  M          reject, then click the 4 corners yourself
             (top-left, top-right, bottom-right, bottom-left)

Results: vendor/Good-Badminton/outputs/demo/

CPU only: expect ~20x real time (303 s for the 14 s demo on a 2020
MacBook Air). Trim your own footage before the first run:
    ffmpeg -ss 0 -t 15 -i your_match.mp4 -c copy clip15.mp4
================================================================
EOS
