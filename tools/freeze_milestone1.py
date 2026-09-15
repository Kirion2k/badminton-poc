"""Write the Milestone 1 freeze manifest from the code and artefacts as they stand."""
import dataclasses, hashlib, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, 'src')
from badminton_poc.config import PlayerConfig, AnalysisConfig, FlightConfig

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
    }

manifest = {
    '_what': (
        'Milestone 1 frozen as an upstream dependency for Milestone 2. The detector settings '
        'below are the tuning surface of Milestone 1; tests/test_milestone1_frozen.py asserts '
        'they still hold these values, so retuning Milestone 1 fails the suite rather than '
        'passing unnoticed.'
    ),
    '_metrics_note': (
        'Metrics are frozen only for clips whose labels are final. test-video3 still carries '
        'model-produced labels; when it is labelled by hand its metrics will move, and that is a '
        'change of labelling, not of the detector. Re-run the freeze script and record the new '
        'figures rather than tuning anything to reach them.'
    ),
    'version': 'milestone-1/1.0',
    'frozen_at_commit': head,
    'frozen_on': '2026-09-15',
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
