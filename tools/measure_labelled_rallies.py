"""Measure every hand-labelled rally and write the range each measurement takes across them.

`rallies.LABELLED_RALLY_RANGES` is defined as the range every hand-labelled rally in the project
falls in. It was first taken over ten rallies on two clips and left alone when clip 5 was
labelled, so all four bounds went stale and `play_evidence` measured detected rallies against a
range four of the real ones fell outside. Running this after a clip is labelled is what keeps
the constant matching its own definition; `tests/test_play_evidence.py` fails if it does not.

The inputs are a run's artefacts (players.jsonl, the TrackNet csv), which are regenerable and not
in git, so this is run by hand rather than from a test.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, 'src')
from badminton_poc.parsing import load_ball_csv
from badminton_poc.rallies import LABELLED_RALLY_RANGES, play_evidence

OUT = Path('configs/labelled-rally-measurements.json')
KEYS = list(LABELLED_RALLY_RANGES)


def labelled_clips():
    """Clips carrying hand-labelled rallies, read off the tracked truth files."""
    found = []
    for path in sorted(Path('outputs').glob('*/truth.json')):
        truth = json.loads(path.read_text())
        if truth.get('labelled') and truth.get('rallies'):
            found.append(path.parent.name)
    return found


def measure(clip):
    analysis = json.loads(Path(f'outputs/{clip}/analysis.json').read_text())
    truth = json.loads(Path(f'outputs/{clip}/truth.json').read_text())
    meta = json.loads(Path(f'outputs/{clip}/metadata.json').read_text())
    ball = load_ball_csv(f'outputs/{clip}/tracknet/work_ball.csv')
    players = [json.loads(line) for line in open(f'outputs/{clip}/players.jsonl')]
    fixed = analysis.get('shuttle_fixed_points') or []
    for rally in truth['rallies']:
        evidence = play_evidence(
            ball, players, rally['start_frame'], rally['end_frame'],
            fps=meta['working']['fps'], width=meta['working']['width'], fixed_points=fixed,
        )
        missing = [k for k in KEYS if not isinstance(evidence.get(k), (int, float))]
        if missing:
            raise SystemExit(
                f'{clip} rally {rally["rally"]} could not be measured on {missing}. A bound taken '
                f'over a subset of the labelled rallies would not be the range it claims to be.')
        yield {'clip': clip, 'rally': rally['rally'], 'start_frame': rally['start_frame'],
               'end_frame': rally['end_frame'], **{k: evidence[k] for k in KEYS}}


clips = labelled_clips()
rallies = [row for clip in clips for row in measure(clip)]
bounds = {k: [min(r[k] for r in rallies), max(r[k] for r in rallies)] for k in KEYS}

OUT.write_text(json.dumps({
    '_what': ('Every hand-labelled rally in the project, measured with play_evidence, and the '
              'range each measurement takes across them. rallies.LABELLED_RALLY_RANGES holds '
              'those bounds and tests/test_play_evidence.py asserts the two still agree.'),
    '_not_a_threshold': ('Observed extremes of a small sample, not a decision boundary. Nothing '
                         'suppresses a rally for falling outside them, and a real rally that is '
                         'short, flat or central can sit outside on any axis.'),
    '_regenerate': 'python tools/measure_labelled_rallies.py, then copy bounds into rallies.py',
    'clips': clips,
    'rally_count': len(rallies),
    'rallies': rallies,
    'bounds': bounds,
}, indent=2) + '\n')

print(f'{len(rallies)} rallies from {len(clips)} clips: {", ".join(clips)}')
for key, (low, high) in bounds.items():
    was = LABELLED_RALLY_RANGES[key]
    moved = '' if [round(v, 3) for v in was] == [round(v, 3) for v in (low, high)] else \
        f'   <- rallies.py still says ({was[0]:.3f}, {was[1]:.3f})'
    print(f'  {key:28} ({low:.3f}, {high:.3f}){moved}')
print(f'wrote {OUT}')
