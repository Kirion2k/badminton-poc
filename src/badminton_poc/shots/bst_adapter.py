"""Run the published BST checkpoint on our shot samples, without touching its repository.

The upstream checkout stays an unmodified clone: we put it on ``sys.path``, import its model and
its class list, and build the tensors ourselves. Nothing is written into it, and the weights live
under this project's ``models/`` directory rather than in the place its own training script would
have saved them.

**Which variant, and why.** Of the five architectures, only ``BST_0`` takes no court-coordinate
input. The other four need a camera-to-court homography we do not have and cannot honestly
derive from a hand-drawn court polygon. On upstream's own merged 25-class benchmark at sequence
length 100 that costs four tenths of a point of accuracy:

    BST-0       0.8284 accuracy   0.8041 macro F1
    BST-CG-AP   0.8322 accuracy   0.8097 macro F1

Paying 0.4 points to avoid fabricating a homography is a good trade.

**The tensor contract**, read from ``bst_infer.py`` and ``shuttleset_dataset.py``:

    JnB       (B, T, 2, (17 + 19) * 2)   joints then bones, flattened over joint and coordinate
    shuttle   (B, T, 2)
    video_len (B,)                        real length before padding

Padding is zeros at the end, and ``video_len`` tells the model how much of the sequence is real.
That is upstream's ``make_seq_len_same``, reproduced here rather than called, because its
decimation branch assumes a source frame rate close to its own and ours is double.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .features import ShotFeatures
from .taxonomy import broad_scores, describe, predicted_slot

DEFAULT_REPO = Path("../BST")
DEFAULT_WEIGHT = Path(
    "models/bst/weight/bst_0_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt"
)
SEQ_LEN = 100
N_CLASSES_MERGED = 25


class BSTUnavailable(RuntimeError):
    """The upstream checkout or its weights are not present. Both are fetched, never vendored."""


@dataclass(frozen=True)
class BSTConfig:
    repo: Path = DEFAULT_REPO
    weight: Path = DEFAULT_WEIGHT
    seq_len: int = SEQ_LEN
    n_classes: int = N_CLASSES_MERGED
    pose_style: str = "JnB_bone"
    device: str = "auto"
    restrict_to_known_hitter: bool = True
    """Fold away the classes claiming the player Milestone 1 says did not hit. Milestone 1 is
    right about that 92.7% of the time, which is far better than a coin flip between two halves
    of the class list."""


def _resolve(path: Path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def load_bst(cfg: BSTConfig, base: Path = Path(".")):
    """Import upstream, build ``BST_0`` and load the checkpoint. Returns (model, classes, device)."""
    import torch

    repo = _resolve(cfg.repo, base)
    package = repo / "stroke_classification"
    if not package.is_dir():
        raise BSTUnavailable(
            f"BST checkout not found at {repo}.\n"
            "Clone it beside this project:\n"
            "  git clone https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer BST"
        )
    weight = _resolve(cfg.weight, base)
    if not weight.exists():
        raise BSTUnavailable(
            f"BST weights not found at {weight}.\n"
            "They are published on Google Drive and are fetched, never committed:\n"
            "  https://drive.google.com/drive/folders/1D4172WZDJWPvpJdpaHDhy_cA-s8F-zR5"
        )

    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from model.bst import BST_0
    from preparing_data.shuttleset_dataset import get_bone_pairs, get_merged_stroke_types

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_bones = len(get_bone_pairs())
    extra = {"J_only": 0, "JnB_bone": 1, "JnB_interp": 1, "Jn2B": 2}[cfg.pose_style]
    in_dim = (17 + n_bones * extra) * 2
    net = BST_0(in_dim=in_dim, n_class=cfg.n_classes, seq_len=cfg.seq_len,
                depth_tem=2, depth_inter=1)
    state = torch.load(str(weight), map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval().to(device)
    return net, list(get_merged_stroke_types()), device


def bone_pairs(cfg: BSTConfig, base: Path = Path(".")) -> list[tuple[int, int]]:
    repo = _resolve(cfg.repo, base)
    package = repo / "stroke_classification"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from preparing_data.shuttleset_dataset import get_bone_pairs
    return list(get_bone_pairs())


def create_bones(joints: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Upstream's ``create_bones``: end minus start, zero wherever either end is missing.

    ``joints`` (T, M, J, 2) -> (T, M, B, 2). The zero test is on the coordinate rather than a
    mask because that is upstream's convention and the weights were trained under it.
    """
    bones = []
    for start, end in pairs:
        a = joints[:, :, start, :]
        b = joints[:, :, end, :]
        bones.append(np.where((a != 0.0) & (b != 0.0), b - a, 0.0))
    return np.stack(bones, axis=-2)


def pad_to_seq_len(array: np.ndarray, seq_len: int) -> np.ndarray:
    """Zero-pad, or truncate, the leading time axis. Upstream pads at the end."""
    n = array.shape[0]
    if n == seq_len:
        return array
    if n > seq_len:
        return array[:seq_len]
    pad = [(0, seq_len - n)] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad)


def to_model_inputs(
    features: Sequence[ShotFeatures], pairs: Sequence[tuple[int, int]], seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch our features into the three arrays BST_0 wants."""
    poses, shuttles, lengths = [], [], []
    for f in features:
        joints = f.joints.astype(np.float32)
        bones = create_bones(joints, pairs).astype(np.float32)
        pose = np.concatenate((joints, bones), axis=-2)          # (T, 2, J+B, 2)
        pose = pose.reshape(pose.shape[0], pose.shape[1], -1)    # (T, 2, (J+B)*2)
        lengths.append(min(pose.shape[0], seq_len))
        poses.append(pad_to_seq_len(pose, seq_len))
        shuttles.append(pad_to_seq_len(f.shuttle.astype(np.float32), seq_len))
    return np.stack(poses), np.stack(shuttles), np.asarray(lengths, dtype=np.int64)


def predict(
    features: Sequence[ShotFeatures],
    cfg: BSTConfig | None = None,
    base: Path = Path("."),
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """One prediction record per shot: upstream class and scores, plus the broad reading."""
    import torch

    cfg = cfg or BSTConfig()
    if not features:
        return []
    net, classes, device = load_bst(cfg, base)
    pairs = bone_pairs(cfg, base)
    pose, shuttle, lengths = to_model_inputs(features, pairs, cfg.seq_len)

    started = time.time()
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            p = torch.from_numpy(pose[i:i + batch_size]).to(device)
            s = torch.from_numpy(shuttle[i:i + batch_size]).to(device)
            v = torch.from_numpy(lengths[i:i + batch_size]).to(device)
            logits_all.append(net(p, s, v).cpu())
    logits = torch.cat(logits_all)
    probs = torch.softmax(logits, dim=1).numpy()
    runtime = time.time() - started

    out = []
    for row, feature in zip(probs, features):
        scores = {classes[i]: float(row[i]) for i in range(len(classes))}
        top = max(scores, key=scores.get)
        slot = feature.sample.hitter if cfg.restrict_to_known_hitter else None
        broad = broad_scores(scores, slot=slot)
        broad_top = max(broad, key=broad.get)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        out.append({
            "clip": feature.sample.clip,
            "shot_id": feature.sample.shot_id,
            "hit_frame": feature.sample.hit_frame,
            "hitter": feature.sample.hitter,
            "source_model": "bst",
            "source_variant": "BST_0",
            "source_weight": Path(cfg.weight).name,
            "source_prediction": top,
            "source_prediction_detail": describe(top),
            "source_predicted_slot": predicted_slot(top),
            "source_slot_agrees_with_milestone1": predicted_slot(top) == feature.sample.hitter,
            "source_scores": scores,
            "source_top3": [{"class": k, "score": round(v, 4)} for k, v in ordered[:3]],
            "broad_prediction": broad_top,
            "broad_score": round(broad[broad_top], 4),
            "broad_scores": {k: round(v, 4) for k, v in broad.items()},
            "restricted_to_hitter": bool(cfg.restrict_to_known_hitter),
            "coverage": feature.coverage(),
            "samples_used": int(min(feature.n_samples, cfg.seq_len)),
        })
    if out:
        out[0]["_runtime_seconds_for_batch"] = round(runtime, 3)
    return out
