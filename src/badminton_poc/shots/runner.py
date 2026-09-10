"""Wiring: load one clip's Milestone 1 outputs, build shots, predict, label, score.

Milestone 1's files are read and never written. Everything this milestone produces lands under
``outputs/<clip>/shots/``, so the older outputs stay byte-for-byte what they were.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..config import PipelineConfig
from ..parsing import load_ball_csv, load_jsonl
from . import baseline as baseline_module
from . import truth as truth_module
from .bst_adapter import BSTConfig, BSTUnavailable, predict as bst_predict
from .court import CourtDepth
from .features import FeatureConfig, ShotFeatures, build_features
from .taxonomy import mapping_document
from .windows import ShotSample, ShotWindowConfig, build_samples


@dataclass(frozen=True)
class ShotPaths:
    root: Path
    clip: str

    @property
    def clip_dir(self) -> Path:
        return self.root / self.clip

    @property
    def shots_dir(self) -> Path:
        return self.clip_dir / "shots"

    @property
    def samples(self) -> Path:
        return self.shots_dir / "shot_samples.jsonl"

    @property
    def predictions(self) -> Path:
        return self.shots_dir / "predictions.jsonl"

    @property
    def truth(self) -> Path:
        return self.shots_dir / "ground_truth.json"

    @property
    def evaluation(self) -> Path:
        return self.shots_dir / "evaluation.json"

    @property
    def previews(self) -> Path:
        return self.shots_dir / "previews"

    def report(self, model: str = "bst") -> Path:
        return self.shots_dir / f"errors_{model}.html"

    @property
    def analysis(self) -> Path:
        return self.clip_dir / "analysis.json"

    @property
    def players(self) -> Path:
        return self.clip_dir / "players.jsonl"

    @property
    def metadata(self) -> Path:
        return self.clip_dir / "metadata.json"

    @property
    def work_video(self) -> Path:
        return self.clip_dir / "work.mp4"

    def ball_csv(self) -> Path:
        found = sorted((self.clip_dir / "tracknet").glob("*ball.csv"))
        if not found:
            raise FileNotFoundError(f"no shuttle CSV under {self.clip_dir / 'tracknet'}")
        return found[0]


@dataclass
class ClipShots:
    paths: ShotPaths
    samples: list[ShotSample]
    features: list[ShotFeatures]
    skipped: list[dict[str, Any]]
    depth: CourtDepth | None
    fps: float
    width: int
    height: int


def load_clip(
    clip: str,
    *,
    output_root: Path = Path("outputs"),
    config: PipelineConfig | None = None,
    window_cfg: ShotWindowConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
) -> ClipShots:
    paths = ShotPaths(Path(output_root), clip)
    analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
    working = json.loads(paths.metadata.read_text(encoding="utf-8"))["working"]
    width, height = int(working["width"]), int(working["height"])
    samples, skipped = build_samples(
        analysis, clip=clip, width=width, height=height, cfg=window_cfg,
        last_frame=int(working.get("frame_count", 0)) - 1 or None,
    )
    ball = load_ball_csv(paths.ball_csv())
    players = {int(r["frame"]): r for r in load_jsonl(paths.players)}
    fixed = analysis.get("shuttle_fixed_points") or []
    polygon = config.court.shuttle_polygon if config is not None else None
    features = [
        build_features(s, players, ball, cfg=feature_cfg, fixed_points=fixed,
                       shuttle_polygon=polygon)
        for s in samples
    ]
    depth = CourtDepth.from_roi(config.court) if config is not None else None
    return ClipShots(paths, samples, features, skipped, depth,
                     float(working["fps"]), width, height)


def write_samples(clip_shots: ClipShots) -> Path:
    clip_shots.paths.shots_dir.mkdir(parents=True, exist_ok=True)
    with clip_shots.paths.samples.open("w", encoding="utf-8") as handle:
        for sample, features in zip(clip_shots.samples, clip_shots.features):
            row = sample.to_dict()
            row["coverage"] = {k: round(v, 4) for k, v in features.coverage().items()}
            row["resampling"] = features.provenance
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if clip_shots.skipped:
        (clip_shots.paths.shots_dir / "skipped_hits.json").write_text(
            json.dumps(clip_shots.skipped, indent=2), encoding="utf-8")
    return clip_shots.paths.samples


def run_predictions(
    clip_shots: ClipShots,
    *,
    bst_cfg: BSTConfig | None = None,
    base: Path = Path("."),
    include_bst: bool = True,
) -> list[dict[str, Any]]:
    """Both models over the same samples, so the comparison is like for like."""
    records: list[dict[str, Any]] = []
    rules = baseline_module.predict(clip_shots.features, depth=clip_shots.depth)
    records.extend(rules)
    if include_bst:
        try:
            records.extend(bst_predict(clip_shots.features, bst_cfg, base=base))
        except BSTUnavailable as exc:
            records.append({"source_model": "bst", "unavailable": str(exc)})
    return records


def write_predictions(clip_shots: ClipShots, records: Sequence[dict[str, Any]]) -> Path:
    clip_shots.paths.shots_dir.mkdir(parents=True, exist_ok=True)
    with clip_shots.paths.predictions.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (clip_shots.paths.shots_dir / "taxonomy.json").write_text(
        json.dumps(mapping_document(), indent=2, ensure_ascii=False), encoding="utf-8")
    return clip_shots.paths.predictions


def seed_truth(clip_shots: ClipShots) -> Path:
    """Create or refresh the labelling file without losing a judgement already made."""
    path = clip_shots.paths.truth
    if path.exists():
        existing = truth_module.load(path)
        fresh = truth_module.refresh(existing, clip_shots.samples, clip_shots.fps)
    else:
        fresh = truth_module.empty_truth(clip_shots.paths.clip, clip_shots.samples,
                                         clip_shots.fps)
    return truth_module.save(path, fresh)


def make_previews(
    clip_shots: ClipShots,
    *,
    pad_seconds: float = 0.4,
    width: int = 640,
    only: Sequence[int] | None = None,
    mark_seconds: float = 0.25,
) -> list[Path]:
    """One short clip per shot, cut from the working video with ffmpeg.

    The window runs from the previous contact to the next one, so a preview shows up to three
    hits and the one being labelled is the middle one. That is genuinely ambiguous to watch, so
    the contact is marked: a red border and the word CONTACT are burned on for ``mark_seconds``
    either side of it, and a header names the shot and the hitter throughout.

    Deliberately a few tenths of a second wider than the classifier's window as well, because a
    human needs to see the shuttle arrive and leave to name a stroke.
    """
    video = clip_shots.paths.work_video
    if not video.exists():
        raise FileNotFoundError(f"working video not found at {video}")
    out_dir = clip_shots.paths.previews
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = clip_shots.fps
    made = []
    for sample in clip_shots.samples:
        if only is not None and sample.shot_id not in only:
            continue
        start = max(0.0, (sample.start_frame / fps) - pad_seconds)
        end = (sample.end_frame / fps) + pad_seconds
        # Where the contact falls inside the trimmed clip.
        at = (sample.hit_frame / fps) - start
        lo, hi = max(0.0, at - mark_seconds), at + mark_seconds
        header = (f"shot {sample.shot_id}  rally {sample.rally}  hitter {sample.hitter}"
                  f"  frame {sample.hit_frame}")
        filters = ",".join([
            f"scale={width}:-2",
            f"drawtext=text='{header}':x=8:y=6:fontsize=16:fontcolor=white:box=1:"
            "boxcolor=black@0.55:boxborderw=4",
            f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=6:"
            f"enable='between(t,{lo:.3f},{hi:.3f})'",
            f"drawtext=text='CONTACT — label THIS shot':x=(w-tw)/2:y=h-34:fontsize=20:"
            f"fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=5:"
            f"enable='between(t,{lo:.3f},{hi:.3f})'",
        ])
        target = out_dir / (
            f"shot{sample.shot_id:03d}_r{sample.rally}_f{sample.hit_frame}_{sample.hitter}.mp4"
        )
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
             "-to", f"{end:.3f}", "-i", str(video), "-vf", filters, "-an", str(target)],
            check=True,
        )
        made.append(target)
    return made


def contact_sheet(
    clip_shots: ClipShots,
    shot_id: int,
    *,
    out_dir: Path,
    tiles: int = 6,
    crop: tuple[int, int, int, int] | None = None,
    tile_width: int = 460,
    source: str = "work",
) -> Path:
    """A strip of frames spanning one shot, for labelling by eye without playing video.

    ``source`` selects the working video or the annotated one. The annotated overlay marks where
    the shuttle tracker thinks the shuttle is, which makes a trajectory legible in stills; it
    shows no shot-type prediction, so it cannot bias the label toward a classifier. It can still
    bias toward the *tracker*, so judge the players and the visible shuttle, not the marker.
    """
    sample = next(s for s in clip_shots.samples if s.shot_id == shot_id)
    video = clip_shots.paths.work_video if source == "work" else clip_shots.paths.clip_dir / "annotated.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"shot{shot_id:03d}.png"
    span = max(1, sample.end_frame - sample.start_frame)
    step = max(1, span // max(1, tiles - 1))
    crop_filter = ""
    if crop:
        w, h, x, y = crop
        crop_filter = f"crop={w}:{h}:{x}:{y},"
    expression = (
        f"select=between(n\\,{sample.start_frame}\\,{sample.end_frame})"
        f"*not(mod(n-{sample.start_frame}\\,{step})),"
    )
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", expression + crop_filter + f"scale={tile_width}:-2,tile=1x{tiles}",
         "-vframes", "1", str(target)],
        check=True,
    )
    return target


def score_clip(clip_shots: ClipShots) -> dict[str, Any]:
    """Score every model in ``predictions.jsonl`` against the hand labels."""
    from . import evaluate as evaluate_module

    truth = truth_module.load(clip_shots.paths.truth)
    rows = {r["shot_id"]: r for r in truth_module.labelled_rows(truth)}
    by_model: dict[str, dict[int, dict[str, Any]]] = {}
    with clip_shots.paths.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "shot_id" in record:
                by_model.setdefault(record["source_model"], {})[record["shot_id"]] = record

    results: dict[str, Any] = {}
    for model, predictions in by_model.items():
        records = []
        for shot_id, row in rows.items():
            prediction = predictions.get(shot_id)
            if prediction is None:
                continue
            records.append({
                "shot_id": shot_id,
                "truth": row["shot_type"],
                "prediction": prediction["broad_prediction"],
                "hitter": row["hitter"],
                "broad_scores": prediction.get("broad_scores"),
            })
        result = evaluate_module.evaluate(records)
        result["acceptance"] = evaluate_module.acceptance(result)
        result["most_confused"] = evaluate_module.most_confused(result, 8)
        results[model] = result

    out = {
        "clip": clip_shots.paths.clip,
        "labelled_shots": len(rows),
        "excluded_not_a_real_contact": sum(
            1 for r in truth.get("shots", []) if r.get("not_a_real_contact")),
        "label_summary": truth_module.summarise(truth),
        "labelled_by": truth.get("_labelled_by"),
        "taxonomy_version": truth.get("_taxonomy_version"),
        "models": results,
        "caveat": (
            "One clip, labelled in-session by the same party that built the pipeline, with a "
            "quarter of the shots flagged uncertain. Provisional throughout."
        ),
    }
    clip_shots.paths.evaluation.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


FAILURE_CAUSES = (
    "pose incomplete",
    "shuttle track incomplete",
    "upstream hit timing or attribution",
    "ambiguous label",
    "genuinely similar shot types",
    "far-player perspective",
    "taxonomy gap",
    "domain shift or model failure",
)


def categorise_failure(
    row: dict[str, Any], prediction: dict[str, Any], sample: dict[str, Any]
) -> str:
    """A first guess at why one shot went wrong, from evidence already on disk.

    Ordered most mechanical first: a missing player explains an error better than "the model was
    wrong" does, and saying "domain shift" about a shot whose pose was half absent would be
    guessing.
    """
    coverage = sample.get("coverage") or {}
    if coverage.get("hitter_joints", 1.0) < 0.7:
        return "pose incomplete"
    if coverage.get("shuttle", 1.0) < 0.6:
        return "shuttle track incomplete"
    if row.get("hitter_looks_wrong"):
        return "upstream hit timing or attribution"
    if row.get("uncertain"):
        return "ambiguous label"
    truth_class, predicted = row.get("shot_type"), prediction.get("broad_prediction")
    if {truth_class, predicted} in ({"clear", "lift"}, {"drop", "net"}, {"drive", "net"}):
        return "genuinely similar shot types"
    if "other" in (truth_class, predicted):
        return "taxonomy gap"
    if row.get("hitter") == "far":
        return "far-player perspective"
    return "domain shift or model failure"


def error_report(clip_shots: ClipShots, model: str = "bst") -> Path:
    """A plain page listing every misclassification with the evidence beside it.

    Not a product frontend. It exists so a wrong answer can be looked at without joining four
    JSON files by hand, and it links each row to its preview clip when one has been made.
    """
    truth = truth_module.load(clip_shots.paths.truth)
    rows = {r["shot_id"]: r for r in truth.get("shots", [])}
    samples = {}
    with clip_shots.paths.samples.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            samples[row["shot_id"]] = row
    predictions = {}
    with clip_shots.paths.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source_model") == model and "shot_id" in record:
                predictions[record["shot_id"]] = record

    wrong = []
    for shot_id, prediction in sorted(predictions.items()):
        row = rows.get(shot_id)
        if row is None or row.get("not_a_real_contact") or not row.get("shot_type"):
            continue
        if prediction["broad_prediction"] == row["shot_type"]:
            continue
        sample = samples.get(shot_id, {})
        wrong.append((shot_id, row, prediction, sample,
                      categorise_failure(row, prediction, sample)))

    def cell(text: str) -> str:
        return f"<td>{text}</td>"

    body = []
    for shot_id, row, prediction, sample, cause in wrong:
        coverage = sample.get("coverage") or {}
        top3 = ", ".join(
            f"{t['class']} {t['score']}" for t in (prediction.get("source_top3") or [])
        )
        body.append(
            "<tr>"
            + cell(str(shot_id))
            + cell(f"r{row.get('rally')}")
            + cell(str(row.get("hit_frame")))
            + cell(f"{row.get('hit_time_s')}s")
            + cell(str(row.get("hitter")))
            + cell(f"<b>{row.get('shot_type')}</b>")
            + cell(prediction.get("broad_prediction", ""))
            + cell(str(prediction.get("broad_score", "")))
            + cell(top3)
            + cell(f"{coverage.get('hitter_joints', '')}")
            + cell(f"{coverage.get('shuttle', '')}")
            + cell("yes" if row.get("uncertain") else "")
            + cell("yes" if row.get("hitter_looks_wrong") else "")
            + cell(cause)
            + cell(row.get("notes", ""))
            + "</tr>"
        )

    counts: dict[str, int] = {}
    for *_ , cause in wrong:
        counts[cause] = counts.get(cause, 0) + 1
    tally = "".join(
        f"<tr><td>{cause}</td><td>{count}</td></tr>"
        for cause, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )

    html = f"""<!doctype html><meta charset="utf-8">
<title>{clip_shots.paths.clip} shot errors ({model})</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#111}}
table{{border-collapse:collapse;margin:16px 0;font-size:13px}}
td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}} caption{{text-align:left;font-weight:600;margin-bottom:6px}}
.note{{background:#fffbe6;border:1px solid #f0e0a0;padding:10px;max-width:60em}}
</style>
<h1>{clip_shots.paths.clip}: shots {model} got wrong</h1>
<p class="note">{len(wrong)} of {len(predictions)} predictions disagree with the hand labels.
The labels were made in-session from frame strips, not by an independent expert, and a quarter
of them are flagged uncertain. The failure cause is a first guess from evidence already on disk,
not a verdict.</p>
<table><caption>Likely causes</caption><tr><th>cause</th><th>count</th></tr>{tally}</table>
<table>
<tr><th>shot</th><th>rally</th><th>frame</th><th>time</th><th>hitter</th><th>truth</th>
<th>predicted</th><th>score</th><th>upstream top 3</th><th>pose</th><th>shuttle</th>
<th>uncertain</th><th>hitter?</th><th>likely cause</th><th>notes</th></tr>
{''.join(body)}
</table>
"""
    target = clip_shots.paths.report(model)
    target.write_text(html, encoding="utf-8")
    return target


def confusion_image(clip_shots: ClipShots, model: str = "bst") -> Path:
    """The confusion matrix as a picture, with the class counts on the axis labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = json.loads(clip_shots.paths.evaluation.read_text(encoding="utf-8"))
    matrix = result["models"][model]["confusion"]
    counts = result["models"][model]["class_counts_truth"]
    classes = [c for c in matrix if counts.get(c) or any(matrix[c].values())]
    data = np.array([[matrix[t][p] for p in classes] for t in classes], dtype=float)

    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(classes), 1.0 + 0.8 * len(classes)))
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels([f"{c} ({counts.get(c, 0)})" for c in classes])
    ax.set_xlabel(f"{model} predicted")
    ax.set_ylabel("labelled (count)")
    ax.set_title(f"{clip_shots.paths.clip}: {model}, {result['labelled_shots']} shots")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if data[i, j]:
                ax.text(j, i, int(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black")
    fig.tight_layout()
    target = clip_shots.paths.shots_dir / f"confusion_matrix_{model}.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target
