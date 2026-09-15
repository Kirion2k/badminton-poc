# Milestone 2, Phase A: what BST actually requires, and whether we can run it

Read from the upstream source at commit `fb9b310` (2026-06-05), not from this project's earlier
notes. The upstream clone at `../BST` is pristine and is never modified: it is imported by path.

**Answer to the phase question: yes, official inference reproduces, within a thousandth on every
published figure.**

---

## 1. The contract, read from source

| | value | where it comes from |
|---|---|---|
| inference entry point | `stroke_classification/main_on_shuttleset/bst_infer.py` | the author's own example |
| architecture variants | `BST`, `BST_CG`, `BST_AP`, `BST_CG_AP` in `bst_infer.py`; `BST_0` in `bst_backbone_main.py` | `model/bst.py` defines all five |
| variant we can run | **`BST_0`** | the only checkpoints downloaded are `bst_0_*` |
| class taxonomy (merged) | 25 = `未知球種` + `Top_`×12 + `Bottom_`×12 | `get_merged_stroke_types()` |
| class taxonomy (full) | 35 = `Top_`×17 + `Bottom_`×17 + `未知球種` | `get_stroke_types()` |
| sequence length | 100 for `between_2_hits_with_max_limits`; 30 for the plain sets | `bst_infer.py`, and the weight file names |
| joint layout | **COCO 17** | `get_bone_pairs('coco')`, 19 pairs |
| pose styles | `J_only` (extra 0), `JnB_bone` / `JnB_interp` (1), `Jn2B` (2) | `bst_infer.py` |
| model input dim | `(n_joints + n_bones × extra) × in_channels` = `(17 + 19) × 2` = **72** | `bst_infer.py` |
| PyTorch / environment | Python 3.11, PyTorch 2.4.0+cu121, NumPy 1.26.4 | upstream README badges |

### Array shapes, confirmed by running it

| array | shape | meaning |
|---|---|---|
| `JnB_bone` | `(3499, 100, 2, 36, 2)` | shots, frames, players, joints+bones, xy |
| `pos` | `(3499, 100, 2, 2)` | shots, frames, players, court xy |
| `shuttle` | `(3499, 100, 2)` | shots, frames, xy |
| `videos_len` | `(3499,)` | true length before padding |
| `labels` | `(3499,)` | class index into the 25 |

`bst_infer.py` flattens the last two pose dimensions before the call, so the model sees
`(b, t, 2, 72)`.

### Normalisation, quoted from upstream

- **shuttle** — normalised by the video resolution, each element in `[0, 1]`.
- **joints** — relative coordinates inside the player's bounding box, origin at its top-left,
  normalised by the box's diagonal, and `center_align=True` by default moves the origin to the
  box centre. Each element ends up in roughly `[-0.5, +0.5]`.
- **position** — feet converted from camera to court coordinates, midpoint taken, normalised by
  the court rectangle, `[0, 1]`.

### Temporal clipping, from `make_seq_len_same`

Longer than the target: subsample by `video_len // target + padding_correction`, take the first
`target`, then zero-pad any remainder. Shorter: zero-pad only. **The authors explicitly do not
interpolate** — "Since they have been normalized, we don't interpolate them" — and `videos_len`
carries the true length so the transformer can mask the padding.

### What `BST_0` does and does not take

`BST_0.forward(JnB, shuttle, video_len)`. **It has no player-position input at all.** The
published result table names its extra inputs as "shuttle position", where `BST` and the gated
variants take "player positions + shuttle position". Our checkpoints are the `BST_0` family, so
the `pos` array is loaded by the dataset and then ignored. Phase B should still produce it, since
it costs nothing and the richer variants need it, but nothing in this phase's numbers depends on
it.

---

## 2. Reproduction

Driven by `tools/bst_reference_infer.py`, which imports upstream's own collation, dataset class,
architecture and class list. The authors' own per-shot arrays for the ShuttleSet_merged test
split were collated with their own `collate_npy` at `seq_len=100`, giving 3,499 shots.

| | published, BST-0 merged seq 100 | reproduced, mean of 7 checkpoints |
|---|---|---|
| accuracy | 0.8284 | **0.8294** |
| macro F1 | 0.8041 | **0.8049** |
| top-2 accuracy | 0.9584 | **0.9585** |

Best single checkpoint: `..._merged_6.pt` at 0.8345 accuracy, 0.8125 macro F1. Spread across the
seven runs from 0.8257 to 0.8345. Runtime 1.2-1.5 s for all 3,499 shots on an RTX 3070; the model
is 1.81M parameters.

Published figures are from the authors' `results_and_images/result_table.xlsx`, sheet
"ShuttleSet_merged (100)", row BST-0, which is marked "(Avg)".

### Environment, and the drift from what upstream asks for

| | upstream asks | we ran |
|---|---|---|
| Python | 3.11 | 3.11.16 |
| PyTorch | 2.4.0+cu121 | **2.14.0+cu130** |
| NumPy | 1.26.4 | **2.4.6** |

Two major versions of PyTorch and one of NumPy, and the figures still land within a thousandth.
Recorded because a later discrepancy might otherwise be blamed on it.

### One practical trap, worth keeping

The authors' preprocessed archive is RAR5. The system `7z` is p7zip 16.02, which predates RAR5's
newer compression and **extracts 10,497 silently empty files** — no error, no warning, correct
names and directory structure, zero bytes each. `libarchive`/`bsdtar` reads it correctly. Both
extractions are still on disk under `models/bst/reference/`, and the script fails with this
explanation rather than a shape error if it is pointed at the empty one.

---

## 3. What this settles for the phases that follow

**The joint layout is not a problem.** BST expects COCO 17 with 19 bone pairs, and the pairs in
`get_bone_pairs('coco')` index nose/eyes/ears 0-4, shoulders 5-6, elbows 7-8, wrists 9-10, hips
11-12, knees 13-14, ankles 15-16. That is exactly what Ultralytics pose emits and exactly what
`players.py` already stores, wrists included at 9 and 10. **Phase C's mapping is the identity**,
which is the best case the brief allows for and should be verified rather than assumed once the
adapter exists.

**The model, the weights and this runtime are sound.** Any disagreement on our own footage is
therefore about our footage or our preprocessing, and cannot be attributed to a broken model path.

**Sequence length is 100 samples, not 100 frames.** `make_seq_len_same` subsamples by an integer
stride, so a window is resampled rather than clipped. Phase B's requirement to use time-based
windows rather than a fixed frame count is consistent with this, and the stride behaviour is what
a 33.75 fps clip and a 59.94 fps clip have to be reconciled through.

**Padding is zeros and `videos_len` is load-bearing.** Anything we generate must set it correctly
or the transformer will attend to padding.
