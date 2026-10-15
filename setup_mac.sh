#!/usr/bin/env bash
set -euo pipefail

REPO="https://github.com/yo-WASSUP/Good-Badminton.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" != "1" ]]; then
  die "This is an Intel Mac. Use ./setup_mac_intel.sh instead —
torch 2.5.1 has no Intel-Mac build (the last one is 2.2.2)."
fi

say "1/6  Locating a native arm64 Python 3.12"

PY=""
for candidate in \
    /opt/homebrew/bin/python3.12 \
    /opt/homebrew/opt/python@3.12/bin/python3.12 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
    "$(command -v python3.12 || true)"; do
  [[ -x "$candidate" ]] || continue
  arch="$("$candidate" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "?")"
  if [[ "$arch" == "arm64" ]]; then PY="$candidate"; break; fi
  echo "  skip $candidate (reports $arch — Rosetta/Intel build)"
done

if [[ -z "$PY" ]]; then
  cat >&2 <<'EOS'

No native arm64 Python 3.12 found.

3.12 is a hard ceiling, not a preference: torch 2.5.1, torchvision 0.20.1,
onnxruntime 1.18.1 and numpy<2 all stop publishing macOS arm64 wheels at
cp312. A Python 3.13+ environment will not work.

Install one:
    /opt/homebrew/bin/brew install python@3.12

Then re-run this script.
EOS
  exit 1
fi
echo "  using $PY ($("$PY" --version))"

say "2/6  Checking ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "  found $(ffmpeg -version | head -1)"
else
  echo "  none on PATH; will link the ffmpeg bundled with imageio-ffmpeg"
  echo "  (arm64 build with libx264) into the venv after install."
fi

say "3/6  Fetching upstream"
if [[ -d vendor/Good-Badminton/.git ]]; then
  echo "  already cloned; leaving it alone (git -C vendor/Good-Badminton pull to update)"
else
  mkdir -p vendor
  git clone --depth 1 "$REPO" vendor/Good-Badminton
fi

say "4/6  Building .venv"
if [[ -d .venv ]]; then
  echo "  .venv exists; reusing. Delete it to start clean."
else
  "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip --quiet
echo "  installing dependencies (a few minutes; torch is ~70 MB)"
pip install -r requirements-mac.txt

python - <<'EOS'
import platform, torch, numpy, cv2, onnxruntime as ort
print(f"\n  python  {platform.python_version()} / {platform.machine()}")
print(f"  torch   {torch.__version__}   mps: {torch.backends.mps.is_available()}")
print(f"  numpy   {numpy.__version__}")
print(f"  opencv  {cv2.__version__}")
print(f"  ort     {ort.__version__}   providers: {ort.get_available_providers()}")
if platform.machine() != "arm64":
    raise SystemExit("\nSTOP: this venv is not arm64.")
if not torch.backends.mps.is_available():
    print("\n  WARNING: MPS unavailable — everything will run on CPU.")
EOS

if [[ ! -e .venv/bin/ffmpeg ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv/bin/ffmpeg
  echo "  linked bundled ffmpeg -> .venv/bin/ffmpeg (active whenever the venv is)"
fi

say "5/6  Downloading weights (~91 MB)"
bash scripts/fetch_weights.sh vendor/Good-Badminton/weights

say "6/6  Patching for macOS"
python scripts/patch_for_mac.py --root vendor/Good-Badminton

cat <<'EOS'

================================================================
Setup complete. Smoke test on the bundled demo clip:

    source .venv/bin/activate
    cd vendor/Good-Badminton
    python main.py \
        --video-path videos/demo.mp4 \
        --template-path templates/demo.png \
        --pose-family yolo-pose \
        --display false

A window opens showing auto-detected court corners.
  Enter / Y  accept
  M          reject and click the 4 corners yourself
             (order: top-left, top-right, bottom-right, bottom-left)

Results land in vendor/Good-Badminton/outputs/demo/

To force CPU for a correctness reference:  GB_DEVICE=cpu python main.py ...
================================================================
EOS
