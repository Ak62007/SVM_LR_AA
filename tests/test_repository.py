"""Guard the promised implementation boundaries and equation cross-references."""
from pathlib import Path
import ast
import re

ROOT = Path(__file__).parents[1]


def test_custom_methods_do_not_import_optimizer_libraries():
    for name in ['barrier', 'primal_dual', 'problems', 'metrics', 'linsolve']:
        text = (ROOT / 'src' / f'{name}.py').read_text()
        tree = ast.parse(text)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        assert not any(any(part in x for part in ['cvxpy', 'sklearn', 'optimize', 'baselines']) for x in imports)
        assert re.search(r'\(\d+\)', text)


def test_equation_references_resolve():
    theory = (ROOT / 'theory/theory_notes.md').read_text()
    numbered = {int(n) for n in re.findall(r'\*\*\((\d+)\)\*\*', theory)}
    for path in (ROOT / 'src').glob('*.py'):
        for line in path.read_text().splitlines():
            if '#' in line:
                comment = line.split('#', 1)[1]
                assert {int(n) for n in re.findall(r'\((\d+)\)', comment)} <= numbered


def test_readme_documents_one_command_and_limits():
    text = (ROOT / 'README.md').read_text()
    for phrase in ['run_all.py', '5e-9', 'not run', 'SAGA', 'generalized', 'requirements.lock']:
        assert phrase in text
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if not target.startswith('http') and target not in {'report/report.md', 'report/report.pdf', 'results/figures/02_gap_vs_solves.png'}:
            assert (ROOT / target).exists()
