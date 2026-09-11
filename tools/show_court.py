"""Draw a candidate court over a frame, with every foot position the pose pass ever saw.

Reading a polygon off a coordinate grid is guesswork until you can see what it excludes. The
feet are the thing that matters: a player whose foot point falls outside the polygon is never a
candidate, and no later stage can recover them.

Usage: show_court.py <clip> <config> [frame]

The frame matters. A clip's midpoint is often dead time, and a court with nobody on it cannot
show whether the net line divides the two players. Pass a frame inside a detected rally.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

clip, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
cap = cv2.VideoCapture(f"outputs/{clip}/work.mp4")
frame = int(sys.argv[3]) if len(sys.argv) > 3 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.45)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
ok, img = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"{clip}: could not read frame {frame}")
H, W = img.shape[:2]

for line in open(f"outputs/{clip}/players.jsonl"):
    r = json.loads(line)
    seen = [r.get("near"), r.get("far")] + list(r.get("not_selected") or []) + \
           list(r.get("rejected") or [])
    for p in seen:
        b = p.get("bbox") if p else None
        if not b or b[3] >= r["height"] - 2 or b[1] <= 2:
            continue
        h = b[3] - b[1]
        col = (70, 230, 70) if h > 0.18 * r["height"] else (0, 170, 255)
        cv2.circle(img, (int((b[0] + b[2]) / 2), int(b[3])), 2, col, -1)

poly = [(int(x * W), int(y * H)) for x, y in cfg["court"]["polygon"]]
cv2.polylines(img, [np.array(poly)], True, (255, 80, 255), 3)
(nx1, ny1), (nx2, ny2) = cfg["court"]["net_line"]
cv2.line(img, (int(nx1 * W), int(ny1 * H)), (int(nx2 * W), int(ny2 * H)), (255, 255, 255), 3)
cv2.putText(img, f"net y={ny1:.3f}", (10, int(ny1 * H) - 8), 0, 0.8, (255, 255, 255), 2)
if cfg["court"].get("shuttle_polygon"):
    sp = [(int(x * W), int(y * H)) for x, y in cfg["court"]["shuttle_polygon"]]
    cv2.polylines(img, [np.array(sp)], True, (60, 220, 255), 2)
for i in range(1, 20):
    x = int(W * i / 20); cv2.line(img, (x, 0), (x, H), (90, 90, 90), 1)
    cv2.putText(img, f"{i/20:.2f}", (x + 2, 14), 0, 0.34, (200, 200, 200), 1)
    y = int(H * i / 20); cv2.line(img, (0, y), (W, y), (90, 90, 90), 1)
    cv2.putText(img, f"{i/20:.2f}", (3, y - 3), 0, 0.34, (200, 200, 200), 1)
out = Path("outputs") / clip / "court_fit.png"
cv2.imwrite(str(out), cv2.resize(img, (1500, int(1500 * H / W))))
print(f"{out}  frame {frame}   green = tall boxes, orange = short boxes, magenta = court")
