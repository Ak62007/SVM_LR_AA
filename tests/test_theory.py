from pathlib import Path
import re


def test_numbered_derivations_exist():
    text = (Path(__file__).parents[1] / 'theory/theory_notes.md').read_text()
    assert set(range(1, 29)) <= {int(x) for x in re.findall(r'\*\*\((\d+)\)\*\*', text)}
    for word in ['Slater', 'generalized', 'certificate', 'short-step', 'floating']:
        assert word in text
