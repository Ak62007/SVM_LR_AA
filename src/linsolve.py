"""A ledger for every solve, including failures and certificate least squares."""
from dataclasses import dataclass, field
from time import perf_counter
import warnings
import numpy as np
from scipy import linalg


@dataclass
class SolveLedger:
    entries: list = field(default_factory=list)
    last_matrix: np.ndarray | None = None

    @property
    def count(self):
        return len(self.entries)

    @property
    def seconds(self):
        return sum(e['seconds'] for e in self.entries)

    def _attempt(self, matrix, rhs, phase, mode):
        started = perf_counter()
        entry = dict(phase=phase, mode=mode, dimension=matrix.shape[0], success=False)
        self.entries.append(entry)  # Count before calling, so failures are included.
        try:
            if mode == 'lstsq':
                value = linalg.lstsq(matrix, rhs, lapack_driver='gelsd', check_finite=True)[0]  # (20), (21)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', linalg.LinAlgWarning)
                    value = linalg.solve(matrix, rhs, assume_a='sym', check_finite=True)  # (20)
            entry['success'] = bool(np.isfinite(value).all())
            if not entry['success']:
                raise FloatingPointError('The linear solution is not finite')
            return value
        finally:
            entry['seconds'] = perf_counter() - started

    def solve(self, matrix, rhs, phase='newton'):
        self.last_matrix = matrix.copy()
        scale = 1. / np.sqrt(np.maximum(np.abs(np.diag(matrix)), np.finfo(float).tiny))  # (20)
        equilibrated = (matrix * scale[:, None]) * scale[None, :]  # (20)
        try:
            v = self._attempt(equilibrated, scale * rhs, phase, 'symmetric')
        except (linalg.LinAlgError, linalg.LinAlgWarning, FloatingPointError):
            v = self._attempt(equilibrated, scale * rhs, phase, 'lstsq')
        return scale * v  # (20)

    def least_squares(self, matrix, rhs, phase='certificate'):
        return self._attempt(matrix, rhs, phase, 'lstsq')  # (21)

    def condition_numbers(self):
        if self.last_matrix is None:
            return float('nan'), float('nan')
        M = self.last_matrix
        scale = 1. / np.sqrt(np.maximum(np.abs(np.diag(M)), np.finfo(float).tiny))  # (20)
        return float(np.linalg.cond(M)), float(np.linalg.cond((M * scale[:, None]) * scale[None, :]))
