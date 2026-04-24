"""Quick-start: extract bboxes from one Halmstad IR video label file.

Usage:
    python examples/halmstad_quickstart.py path/to/IR_DRONE_001_LABELS.mat

Or, if you have the Halmstad dataset locally, just run with no args.
"""
import sys
from pathlib import Path
from mcos_decoder import load_groundtruth

DEFAULT = Path(
    "/home/burak/ir-tracker/publication/drone_detection_dataset/"
    "Drone-detection-dataset-master/Data/Video_IR/IR_DRONE_001_LABELS.mat"
)

path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
if not path.exists():
    print(f"Not found: {path}")
    print("Pass a label .mat path or download the Halmstad dataset.")
    sys.exit(1)

bboxes = load_groundtruth(path)
n_total = len(bboxes)
n_visible = sum(1 for b in bboxes if b is not None)

print(f"File:    {path.name}")
print(f"Frames:  {n_total}")
print(f"Visible: {n_visible} ({100*n_visible/n_total:.1f}%)")
print()
print("First 5 frames:")
for i, bb in enumerate(bboxes[:5], start=1):
    if bb is None:
        print(f"  frame {i}: target absent")
    else:
        x, y, w, h = bb
        print(f"  frame {i}: bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
