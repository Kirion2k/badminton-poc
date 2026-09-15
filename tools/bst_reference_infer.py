"""Phase A: reproduce official BST inference on the authors' own data, before adapting ours.

Nothing here adapts our pipeline. The point is to establish that the model, the weights and this
machine's runtime agree with the published numbers, so that a later disagreement on our footage
can be attributed to our footage rather than to a broken model path.

The upstream repository is imported by path and never modified or copied into this one. Its own
collation, dataset class, architecture and class list do the work; this file only drives them and
records what came out.

    python tools/bst_reference_infer.py --limit 1      # smoke test, one checkpoint
    python tools/bst_reference_infer.py                # all merged seq-100 checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT.parent / "BST" / "stroke_classification"
REFERENCE = ROOT / "models" / "bst" / "reference"
WEIGHTS = ROOT / "models" / "bst" / "weight"

sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "preparing_data"))

from model.bst import BST_0  # noqa: E402
from preparing_data.shuttleset_dataset import (  # noqa: E402
    Dataset_npy_collated,
    get_bone_pairs,
    get_merged_stroke_types,
)
from preparing_data.prepare_train_on_shuttleset_merged import collate_npy  # noqa: E402

# Read from the authors' own result table, sheet "ShuttleSet_merged (100)", row BST-0.
PUBLISHED = {"accuracy": 0.8284, "macro_f1": 0.8041, "min_class_f1": 0.5822, "top2": 0.9584}


def metrics(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    pred = logits.argmax(axis=1)
    order = np.argsort(-logits, axis=1)
    top2 = float(np.mean([labels[i] in order[i, :2] for i in range(len(labels))]))
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (labels == c)))
        fp = int(np.sum((pred == c) & (labels != c)))
        fn = int(np.sum((pred != c) & (labels == c)))
        if tp + fn == 0:          # class absent from the split: not scored
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "accuracy": round(float(np.mean(pred == labels)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "min_class_f1": round(float(np.min(f1s)), 4),
        "top2_accuracy": round(top2, 4),
        "classes_present": len(f1s),
    }


def ensure_collated(seq_len: int) -> Path:
    """Run the authors' own collation over their own per-shot arrays, once."""
    raw = REFERENCE / "extracted" / "dataset_npy_between_2_hits_with_max_limits"
    out = REFERENCE / f"collated_seq_{seq_len}"
    if (out / "test" / "labels.npy").exists():
        return out
    if not (raw / "test").is_dir():
        raise SystemExit(
            f"{raw/'test'} is missing. The authors' archive extracts to zero-byte files under "
            "p7zip 16.02, which predates RAR5; use libarchive (bsdtar) instead."
        )
    print(f"collating {raw/'test'} -> {out} (once)")
    collate_npy(root_dir=raw, set_name="test", seq_len=seq_len, save_dir=out)
    return out


@torch.no_grad()
def run(net, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    net.eval()
    logits_all, labels_all = [], []
    started = time.perf_counter()
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose = human_pose.to(device)
        shuttle = shuttle.to(device)
        video_len = video_len.to(device)
        # Exactly as bst_infer.py does it: flatten (J+B, d) into the model's in_dim.
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        # BST_0 is the backbone variant: pose and shuttle only, no player positions.
        logits = net(human_pose, shuttle, video_len)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(logits_all), np.concatenate(labels_all), time.perf_counter() - started


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--pose-style", default="JnB_bone")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="only this many checkpoints")
    ap.add_argument("--out", default="outputs/bst_reference/phase_a.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = get_merged_stroke_types()
    n_bones = len(get_bone_pairs())
    extra = {"J_only": 0, "JnB_interp": 1, "JnB_bone": 1, "Jn2B": 2}[args.pose_style]
    n_joints, in_channels = 17, 2
    in_dim = (n_joints + n_bones * extra) * in_channels

    collated = ensure_collated(args.seq_len)
    dataset = Dataset_npy_collated(collated, "test", args.pose_style)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    shapes = {
        "human_pose": list(dataset.human_pose.shape),
        "pos": list(dataset.pos.shape),
        "shuttle": list(dataset.shuttle.shape),
        "videos_len": list(dataset.videos_len.shape),
        "labels": list(dataset.labels.shape),
        "model_in_dim": in_dim,
        "n_joints": n_joints,
        "n_bones": n_bones,
        "n_classes": len(classes),
    }
    print(json.dumps(shapes, indent=1))

    ckpts = sorted(WEIGHTS.glob(
        f"bst_0_{args.pose_style}_between_2_hits_with_max_limits_seq_{args.seq_len}_merged*.pt"))
    if args.limit:
        ckpts = ckpts[:args.limit]
    if not ckpts:
        raise SystemExit(f"no matching checkpoints in {WEIGHTS}")

    results = []
    for ckpt in ckpts:
        net = BST_0(in_dim=in_dim, n_class=len(classes), seq_len=args.seq_len,
                    depth_tem=2, depth_inter=1).to(device)
        net.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
        params = sum(p.numel() for p in net.parameters())
        logits, labels, seconds = run(net, loader, device)
        m = metrics(logits, labels, len(classes))
        m.update(checkpoint=ckpt.name, seconds=round(seconds, 2), parameters=params)
        results.append(m)
        print(f"  {ckpt.name:70} acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
              f"top2 {m['top2_accuracy']:.4f}  {seconds:.1f}s")

    acc = [r["accuracy"] for r in results]
    f1 = [r["macro_f1"] for r in results]
    top2 = [r["top2_accuracy"] for r in results]
    summary = {
        "_what": "Phase A: official BST-0 inference on the authors' own ShuttleSet_merged test "
                 "split, reproduced before any of our data is adapted to it.",
        "upstream_commit": "fb9b310",
        "architecture": "BST_0 (the backbone variant: pose and shuttle only, no player "
                        "positions). The checkpoints available are bst_0_*; the example in "
                        "bst_infer.py names BST_CG_AP weights we do not have.",
        "pose_style": args.pose_style,
        "seq_len": args.seq_len,
        "shapes": shapes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "upstream_asks_for": "Python 3.11, PyTorch 2.4.0+cu121, NumPy 1.26.4",
        },
        "published_bst0_merged_seq100": PUBLISHED,
        "reproduced": {
            "n_checkpoints": len(results),
            "accuracy_mean": round(float(np.mean(acc)), 4),
            "accuracy_best": round(float(np.max(acc)), 4),
            "macro_f1_mean": round(float(np.mean(f1)), 4),
            "top2_mean": round(float(np.mean(top2)), 4),
        },
        "per_checkpoint": results,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\npublished  acc {PUBLISHED['accuracy']:.4f}  macroF1 {PUBLISHED['macro_f1']:.4f}  "
          f"top2 {PUBLISHED['top2']:.4f}")
    print(f"reproduced acc {summary['reproduced']['accuracy_mean']:.4f}  "
          f"macroF1 {summary['reproduced']['macro_f1_mean']:.4f}  "
          f"top2 {summary['reproduced']['top2_mean']:.4f}   (mean of {len(results)})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
