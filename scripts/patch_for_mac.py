#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(HERE), "vendor", "Good-Badminton")

DEVICE_MODULE_PATH = os.path.join("badminton_analysis", "device_select.py")

DEVICE_MODULE = '''import os
import platform
import sys


def _override():
    value = os.environ.get("GB_DEVICE", "").strip().lower()
    return value if value in ("cpu", "mps", "cuda") else None


def _mps_available():
    if sys.platform == "darwin" and platform.machine() == "x86_64":
        return False
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pick_torch_device():
    forced = _override()
    if forced == "cuda":
        return 0
    if forced:
        return forced
    if _cuda_available():
        return 0
    if _mps_available():
        return "mps"
    return "cpu"


def pick_ort_device():
    forced = _override()
    if forced:
        return forced
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"
'''

SHUTTLE = os.path.join("badminton_analysis", "detection", "shuttlecock.py")
YOLO_POSE = os.path.join("badminton_analysis", "detection", "yolo_pose.py")
RTMPOSE = os.path.join("badminton_analysis", "detection", "rtmpose.py")
SYSTEM = os.path.join("badminton_analysis", "system.py")
VIDEO_AUDIO = os.path.join("badminton_analysis", "media", "video_audio.py")

TK_MARKER = ("# tkinter is imported lazily in _get_template_path "
             "(patch_for_mac.py: Homebrew Python has no tkinter)\n")

PATCHES = [
    (
        SHUTTLE,
        '        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():\n'
        '            self.ultra_device = 0\n'
        '        else:\n'
        '            self.ultra_device = "cpu"\n',
        '        from ..device_select import pick_torch_device\n'
        '        self.ultra_device = pick_torch_device()\n',
    ),
    (
        YOLO_POSE,
        '        if device in (None, "auto"):\n'
        '            selected = "cpu"\n'
        '            try:\n'
        '                import torch\n'
        '                if torch.cuda.is_available():\n'
        '                    selected = 0\n'
        '            except Exception:\n'
        '                selected = "cpu"\n'
        '            self.device = selected\n'
        '        else:\n'
        '            self.device = device\n',
        '        if device in (None, "auto"):\n'
        '            from ..device_select import pick_torch_device\n'
        '            self.device = pick_torch_device()\n'
        '        else:\n'
        '            self.device = device\n',
    ),
    (
        RTMPOSE,
        "        if device in (None, 'auto'):\n"
        "            selected = 'cpu'\n"
        "            try:\n"
        "                import torch  # noqa: F401\n"
        "                # noinspection PyUnresolvedReferences\n"
        "                if torch.cuda.is_available():\n"
        "                    selected = 'cuda'\n"
        "            except Exception:\n"
        "                selected = 'cpu'\n"
        "            self.device = selected\n"
        "        else:\n"
        "            self.device = device\n",
        "        if device in (None, 'auto'):\n"
        "            from ..device_select import pick_ort_device\n"
        "            self.device = pick_ort_device()\n"
        "        else:\n"
        "            self.device = device\n",
    ),
    (
        SYSTEM,
        "from tkinter import filedialog\nimport tkinter as tk\n",
        TK_MARKER,
    ),
    (
        SYSTEM,
        "        try:\n"
        "            root = tk.Tk()\n",
        "        try:\n"
        "            import tkinter as tk\n"
        "            from tkinter import filedialog\n"
        "            root = tk.Tk()\n",
    ),
    (
        VIDEO_AUDIO,
        "    result = subprocess.run(command, capture_output=True, text=True, timeout=180)\n",
        "    # timeout removed by patch_for_mac.py: a long clip on a slow CPU can take\n"
        "    # >180 s to encode, and hitting it discards the final MP4.\n"
        "    result = subprocess.run(command, capture_output=True, text=True, timeout=None)\n",
    ),
]

MAIN = "main.py"
MAPPER = os.path.join("badminton_analysis", "court", "mapper.py")
COURT_TRAJ = os.path.join("badminton_analysis", "visualization", "court_trajectory.py")
WEBUI_APP = os.path.join("webui", "app.py")
WEBUI_PIPE = os.path.join("webui", "pipeline.py")

ENGLISH = [
    (MAIN, "description='羽毛球比赛视频分析系统'", "description='Badminton match video analysis'"),
    (MAIN, "help='输入视频文件路径'", "help='Input video path'"),
    (MAIN, "help='球场模板图像路径；不提供时会弹出文件选择框'",
     "help='Court template image (a frame showing the whole court); a file picker opens if omitted'"),
    (MAIN, "help='输出目录，默认 outputs/<视频文件名>'", "help='Output directory (default outputs/<video name>)'"),
    (MAIN, "help='YOLO 羽毛球检测模型路径'", "help='YOLO shuttlecock detector weights'"),
    (MAIN, "help='姿态模型族'", "help='Pose model family'"),
    (MAIN, "help='RTMPose / RTMO 模型档位'",
     "help='RTMPose/RTMO size (offline: rtmpose=balanced, rtmo=lightweight only)'"),
    (MAIN, "help='YOLO pose 模型路径或模型名'", "help='YOLO pose weights path or model name'"),
    (MAIN, "help='是否显示姿态检测 ROI 框，默认 true'", "help='Draw the pose-detection ROI box (default true)'"),
    (MAIN, "help='是否显示视频窗口，默认 true'", "help='Show a live preview window (default true)'"),
    (MAIN, "help='是否显示人体骨架，默认 true'", "help='Draw skeletons (default true)'"),
    (MAIN, "help='是否显示球员轨迹，默认 true'", "help='Draw player trajectories (default true)'"),
    (MAIN, "help='是否显示球场轨迹，默认 true'", "help='Draw the mini-court trajectory overlay (default true)'"),
    (MAIN, "help='是否显示羽毛球轨迹，默认 true'", "help='Draw the shuttlecock trail (default true)'"),
    (MAIN, "help='是否显示球员统计信息，默认 true'", "help='Draw the player stats panel (default true)'"),
    (MAIN, "help='保存处理后的图像'", "help='Save every processed frame as PNG'"),
    (MAIN, "help='显示性能统计信息'", "help='Print per-stage timings'"),
    (MAIN, "help='是否生成球员位置热力图和散点图，默认 true'",
     "help='Generate position heatmaps and scatter plots (default true)'"),
    (MAIN, "help='是否保留原视频音频，默认 true'", "help='Keep the original audio (default true)'"),
    (MAIN, "default='zh', choices=['zh', 'en'], help='选择界面语言 (zh/en)'",
     "default='en', choices=['zh', 'en'], help='Overlay and plot language (default en)'"),
    (MAIN, 'print("\\n开始生成球员位置可视化...")', 'print("\\nGenerating player position plots...")'),
    (MAIN, 'print("球员位置可视化完成")', 'print("Player position plots done")'),
    (SYSTEM, 'print(f"\\n处理完成:")', 'print(f"\\nProcessing complete:")'),
    (SYSTEM, 'print(f"原始视频时长: {video_duration:.2f} 秒")',
     'print(f"Video duration:  {video_duration:.2f} s")'),
    (SYSTEM, 'print(f"处理耗时: {processing_time:.2f} 秒")',
     'print(f"Processing time: {processing_time:.2f} s")'),
    (SYSTEM, 'print(f"处理速度比: {processing_time/video_duration:.2f}x")',
     'print(f"Slower than real time by {processing_time/video_duration:.2f}x")'),
    (MAPPER, 'print("请按顺序点击球场 4 个角点：左上、右上、右下、左下。")',
     'print("Click the 4 court corners in order: top-left, top-right, bottom-right, bottom-left.")'),
    (MAPPER, 'print("窗口图片顶部会显示点击顺序；ROI 会自动生成，不需要手动标注。")',
     'print("The click order is shown at the top of the window; the ROI is generated automatically.")'),
    (MAPPER, 'print("已自动生成 ROI（球员姿态检测范围），无须手动选择。")',
     'print("ROI (pose detection area) generated automatically.")'),
    (MAPPER, 'print("未完成球场范围标注，请重试。")',
     'print("Court annotation incomplete - please try again.")'),
    (COURT_TRAJ, 'print(f"绘制球场轨迹出错: {e}")', 'print(f"Court trajectory drawing failed: {e}")'),
    (WEBUI_APP, '    t = _UI_TEXT["zh"]\n', '    t = _UI_TEXT["en"]\n'),
    (WEBUI_APP, 'value="zh", label=t["language"],', 'value="en", label=t["language"],'),
    (WEBUI_PIPE, 'language = options.get("language", "zh")', 'language = options.get("language", "en")'),
]
PATCHES += ENGLISH


def read(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def grouped():
    groups = OrderedDict()
    for rel, old, new in PATCHES:
        groups.setdefault(rel, []).append((old, new))
    return groups


def revert(root):
    changed = 0
    for rel in grouped():
        path = os.path.join(root, rel)
        backup = path + ".orig"
        if os.path.exists(backup):
            shutil.copyfile(backup, path)
            os.remove(backup)
            print(f"  reverted {rel}")
            changed += 1
    module = os.path.join(root, DEVICE_MODULE_PATH)
    if os.path.exists(module):
        os.remove(module)
        print(f"  removed   {DEVICE_MODULE_PATH}")
        changed += 1
    print(f"\n{changed} file(s) restored." if changed else "\nNothing to revert.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Patch Good-Badminton for macOS. Idempotent; --revert undoes.")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Good-Badminton checkout (default: %(default)s)")
    parser.add_argument("--check", action="store_true", help="report without writing")
    parser.add_argument("--revert", action="store_true", help="restore .orig backups")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(os.path.join(root, "badminton_analysis")):
        print(f"ERROR: no badminton_analysis/ under {root}", file=sys.stderr)
        print("Pass --root <path to Good-Badminton checkout>.", file=sys.stderr)
        return 2

    print(f"Target: {root}\n")
    if args.revert:
        return revert(root)

    writes, problems = [], 0
    for rel, edits in grouped().items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  [WARN]  {rel}: file not found")
            problems += 1
            continue
        original = read(path)
        text = original
        applied = 0
        for old, new in edits:
            if new in text:
                continue
            if old in text:
                text = text.replace(old, new, 1)
                applied += 1
            else:
                print(f"  [WARN]  {rel}: expected code block not found - upstream changed?")
                problems += 1
        if applied:
            print(f"  [apply] {rel} ({applied} edit{'s' if applied > 1 else ''})")
            writes.append((path, original, text))
        else:
            print(f"  [ok]    {rel} already patched")

    module_path = os.path.join(root, DEVICE_MODULE_PATH)
    module_needed = not os.path.exists(module_path)
    if module_needed:
        print(f"  [apply] {DEVICE_MODULE_PATH} (new file)")

    if args.check:
        print("\n--check: nothing written.")
        return 1 if problems else 0

    if module_needed:
        write(module_path, DEVICE_MODULE)
    for path, original, text in writes:
        backup = path + ".orig"
        if not os.path.exists(backup):
            write(backup, original)
        write(path, text)

    print(f"\nPatched {len(writes)} file(s).")
    if problems:
        print("WARNING: some blocks were not found. Re-read those files by hand;")
        print("they may still crash without tkinter or run on CPU only.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
