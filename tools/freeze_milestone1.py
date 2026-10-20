"""Write the Milestone 1 freeze manifest from the code and artefacts as they stand."""
import dataclasses, hashlib, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, 'src')
from badminton_poc.config import PlayerConfig, AnalysisConfig, FlightConfig
from badminton_poc.truth import find_impossible_alternations

def defaults(cls):
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
    return out

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()

clips = {}
for clip, provenance in (('test-video2', 'human'), ('test-video3', 'human'), ('test-video5', 'human')):
    b = json.loads(Path(f'outputs/{clip}/baseline.json').read_text())
    truth = json.loads(Path(f'outputs/{clip}/truth.json').read_text())
    clips[clip] = {
        'label_provenance': provenance,
        'config_sha256_16': sha(f'configs/{clip}.json'),
        'truth_sha256_16': sha(f'outputs/{clip}/truth.json'),
        'labelled_contacts': len(truth.get('contacts') or []),
        'labelled_rallies': len(truth.get('rallies') or []),
        'metrics': {
            'hit_precision': b['hit_detection']['precision'],
            'hit_recall': b['hit_detection']['recall'],
            'hit_f1': b['hit_detection']['f1'],
            'hitter_accuracy': b['hitter_assignment']['accuracy'],
            'hitter_scored': b['hitter_assignment']['scored'],
            'rallies_recovered': b['rally_segmentation']['recovered'],
            'rallies_labelled': b['rally_segmentation']['labelled_rallies'],
            'rallies_spurious': b['rally_segmentation']['spurious'],
            'coverage': b['player_tracking']['both_players_fraction'],
            'coverage_trusted': b['player_tracking']['coverage_trusted'],
        },
        'metrics_are_frozen': provenance == 'human',
        # Carried per clip because it is the one label-quality figure that does not come from
        # the detector, so a clip whose labels were confirmed rather than given cannot pass
        # unremarked just because its accuracy reads well.
        'impossible_alternations': len(find_impossible_alternations(truth)),
    }

manifest = {
    '_what': (
        'Milestone 1 frozen as an upstream dependency for Milestone 2. The detector settings '
        'below are the tuning surface of Milestone 1; tests/test_milestone1_frozen.py asserts '
        'they still hold these values, so retuning Milestone 1 fails the suite rather than '
        'passing unnoticed.'
    ),
    '_metrics_note': (
        'All three clips now carry hand labels, so all three sets of metrics are frozen. A '
        'figure here moves only when the labelling changes, and that is a change of labelling, '
        'not of the detector: re-run the freeze script and record the new figures rather than '
        'tuning anything to reach them.'
    ),
    '_detector_change_note': (
        'milestone-1/1.3 (2026-10-20) is the first re-freeze caused by the detector rather than '
        'the labels. FlightConfig gained reentry_kink_speed_ratio: a racket contact made as the '
        'shuttle came back into view after the tracker lost it, found as a sudden jump in step '
        'against the approach it was seen making. It recovers clip 2 at 4317 and clip 3 at 971, '
        'both correctly attributed, adds no false positive on any labelled frame, and moves no '
        'clip 5 figure. The evidence is two positives against eight blind re-entries that read '
        '2.3x or less; see the field docstring.'
    ),
    '_label_quality_note': (
        'Not every hand labelling was produced the same way, and as of 2026-10-08 the '
        'impossible_alternations count no longer shows that up. Clips 3 and 5 were labelled '
        'blind from previews carrying no model output and reached zero on their own. Clip 2 '
        "reached zero by repair: its contacts were migrated from candidate verdicts whose "
        'true_hitter agreed with detected_hitter on 60 of 61 real hits, which is what confirming '
        'a detector looks like, and it held thirteen impossible alternations. Four hitters were '
        'corrected by the owner on 2026-09-16, four more were resolved model-derived the same '
        'day, and the last two labels (frames 7227 and 7248) were removed on 2026-10-08 as '
        'tracker artefacts, with the frames and the reasoning kept in the truth file under '
        '_removed_contacts. A zero for clip 2 therefore means repaired, not independently '
        'labelled, and its metrics remain the least trustworthy of the three.'
    ),
    'version': 'milestone-1/1.3',
    'frozen_at_commit': head,
    'frozen_on': '2026-10-20',
    'detector_defaults': {
        'PlayerConfig': defaults(PlayerConfig),
        'AnalysisConfig': defaults(AnalysisConfig),
        'FlightConfig': defaults(FlightConfig),
    },
    'clips': clips,
}
Path('configs/milestone-1-frozen.json').write_text(json.dumps(manifest, indent=1) + '\n')
print('wrote configs/milestone-1-frozen.json')
print(' frozen knobs:', sum(len(v) for v in manifest['detector_defaults'].values()))
for c, v in clips.items():
    print(f"  {c}: labels={v['label_provenance']}, metrics frozen={v['metrics_are_frozen']}")
