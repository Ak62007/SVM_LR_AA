"""Static, exportable figures. Quantities follow theory equations (6), (23)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'barrier': '#167d9a', 'primal_dual': '#d35b35', 'CLARABEL': '#7057a3',
          'SCS': '#b3851f', 'sklearn': '#537b44'}
plt.rcParams.update({'figure.dpi': 125, 'savefig.dpi': 160, 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .2, 'lines.linewidth': 1.8,
                     'legend.frameon': False})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, .95) if fig._suptitle else None)
    fig.savefig(directory / f'{name}.png', bbox_inches='tight')
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def convergence(results, output):
    kinds = [kind for kind in ['svm', 'logreg'] if any(k == kind for k, method in results)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.1 * len(kinds), 4.2), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for method in ['barrier', 'primal_dual']:
            r = results[kind, method]
            x = [h['solves'] for h in r.history]
            key = 'barrier_gap' if method == 'barrier' else 'surrogate_gap'
            y = [h[key] for h in r.history]
            ax.semilogy(x, y, label='barrier (m/tau)' if method == 'barrier' else 'primal-dual (surrogate)', color=COLORS[method])
            if np.isfinite(r.certified_gap):
                ax.scatter([r.solves], [r.certified_gap if r.certified_gap > 0 else 1e-16],
                           marker='x', color=COLORS[method], s=45)
        ax.axhline(1e-8, color='0.4', linestyle=':', linewidth=1.)
        ax.set(xlabel='Cumulative linear-system solves', ylabel='Gap (log scale)', title=kind.upper())
        ax.legend()
    fig.suptitle('Nominal barrier bound and primal-dual surrogate gap; x = recovered certificate', y=.995, fontsize=12)
    save(fig, output, '02_gap_vs_solves')
    fig, axes = plt.subplots(len(kinds), 3, figsize=(15, 4 * len(kinds)), squeeze=False)
    for row, kind in enumerate(kinds):
        for method in ['barrier', 'primal_dual']:
            r = results[kind, method]
            x = np.arange(len(r.history))
            for col, key, title in [(0, 'dual_residual', 'Raw dual residual'),
                                    (1, 'complementarity_residual', 'Centering residual'),
                                    (2, 'objective_error', 'Absolute objective error vs. reference')]:
                values = np.maximum([abs(h[key]) for h in r.history], 1e-16)
                axes[row, col].semilogy(x, values, label=method, color=COLORS[method])  # (23)
                axes[row, col].set(title=f'{kind}: {title}', xlabel='Logged Newton state', ylabel=title)
                axes[row, col].legend()
    save(fig, output, '02_residuals_and_objective_error')
    fig, axes = plt.subplots(2, len(kinds), figsize=(6.1 * len(kinds), 7.5), squeeze=False)
    for col, kind in enumerate(kinds):
        for method in ['barrier', 'primal_dual']:
            r = results[kind, method]
            key = 'barrier_gap' if method == 'barrier' else 'surrogate_gap'
            x = np.arange(len(r.history))
            axes[0, col].semilogy(x, [h[key] for h in r.history], label=method, color=COLORS[method])
            for residual, style in [('r_pri', '-'), ('primal_residual', ':')]:
                axes[1, col].semilogy(x, np.maximum([h[residual] for h in r.history], 1e-18),
                                     style, label=f'{method}: {residual}', color=COLORS[method])
        axes[0, col].set(title=kind.upper(), xlabel='Logged Newton state', ylabel='Gap')
        axes[1, col].set(xlabel='Logged Newton state', ylabel='Primal residual (zero shown at 1e-18)')
        axes[0, col].legend(fontsize=8)
        axes[1, col].legend(fontsize=7)
    save(fig, output, '02_gap_vs_iteration_and_primal')



def tuning(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, group in frame.groupby('problem'):
        g = group.sort_values('mu')
        axes[0].plot(g.mu, g.newton_iterations, 'o-', label=kind)
        axes[1].plot(g.mu, g.outer_iterations, 'o-', label=kind)
    axes[0].set(ylabel='Newton directions, including center checks', xlabel='Barrier growth factor mu', xscale='log')
    axes[1].set(ylabel='Outer centers', xlabel='Barrier growth factor mu', xscale='log')
    for ax in axes:
        ax.legend()
    save(fig, output, '03_barrier_growth')


def comparison(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, kind in zip(axes, ['svm', 'logreg']):
        g = frame[(frame.problem == kind) & (frame.dataset == 'synthetic')]
        ax.bar(g.method, g.solve_time, color=[COLORS[m] for m in g.method])
        ax.tick_params(axis='x', rotation=25)
        ax.set(yscale='log', ylabel='Median solve time (seconds)', title=kind.upper())
    save(fig, output, '04_solver_times')


def sensitivity(frame, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, kind in enumerate(['svm', 'logreg']):
        for col, experiment in enumerate(['feature_scales', 'correlation']):
            ax = axes[row, col]
            subset = frame[(frame.problem == kind) & (frame.experiment == experiment)]
            for (method, scaling), g in subset.groupby(['method', 'scaling']):
                g = g.sort_values('knob')
                style = '--' if scaling == 'standardized' else '-'
                ax.plot(g.knob, g.newton_iterations, style, marker='o', color=COLORS[method],
                        label=f'{method}, {scaling}')
                bad = g.status != 'converged'
                ax.scatter(g.loc[bad, 'knob'], g.loc[bad, 'newton_iterations'], marker='x', s=90,
                           color=COLORS[method], linewidths=2.5)
            ax.set(title=f'{kind}: {experiment.replace("_", " ")}', xlabel='Scale span (decades)' if col == 0 else 'Target cond(X.T X)',
                   ylabel='Newton directions; x = failure')
            if col == 1:
                ax.set_xscale('log')
            ax.legend(fontsize=7)
    save(fig, output, '05_conditioning_iterations')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, kind in zip(axes, ['svm', 'logreg']):
        g = frame[frame.problem == kind]
        for method, group in g.groupby('method'):
            ax.scatter(group.last_condition, np.maximum(group.dual_residual, 1e-16),
                       label=method, color=COLORS[method], alpha=.7)
        ax.set(xscale='log', yscale='log', xlabel='Last Newton matrix condition number',
               ylabel='Final recovered/raw KKT dual residual', title=kind.upper())
        ax.legend()
    save(fig, output, '05_matrix_condition_and_residual')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, kind in zip(axes, ['svm', 'logreg']):
        g = frame[(frame.problem == kind) & (frame.experiment == 'regularization')]
        for method, group in g.groupby('method'):
            group = group.sort_values('knob')
            ax.semilogx(group.knob, group.newton_iterations, 'o-', label=method, color=COLORS[method])
        ax.set(title=kind.upper(), xlabel='C (SVM) or lambda (logistic)', ylabel='Newton directions')
        ax.legend()
    save(fig, output, '05_regularization')



def sizes(frame, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for row, kind in enumerate(['svm', 'logreg']):
        for col, variable in enumerate(['n', 'd']):
            ax = axes[row, col]
            subset = frame[(frame.sweep == variable) & (frame.problem == kind) & frame.solve_time.notna()]
            for (method, structure), g in subset.groupby(['method', 'structure']):
                g = g.sort_values(variable)
                ax.loglog(g[variable], g.solve_time, 'o--' if structure == 'full' else 'o-',
                          label=f'{method}, {structure}', color=COLORS[method])
                bad = g.status != 'converged'
                ax.scatter(g.loc[bad, variable], g.loc[bad, 'solve_time'], marker='x', s=90,
                           color=COLORS[method], linewidths=2.5)
            ax.set(title=kind.upper(), xlabel=f'Number of {"samples" if variable == "n" else "features"} ({variable})',
                   ylabel='Median solve time (seconds); x = failure')
            ax.legend(fontsize=9)
    save(fig, output, '06_problem_size')


def self_concordance(frame, analytic, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (kind, outer), g in frame.groupby(['problem', 'outer']):
        if outer > 2:
            continue
        axes[0].semilogy(g.inner, np.maximum(g.decrement, 1e-16), 'o-', markersize=3,
                         label=f'{kind}, outer {outer}')
    axes[0].axhline(.25, color='black', linestyle=':', label='delta=1/4: local quadratic region')  # (10)
    axes[0].set(xlabel='Inner Newton direction', ylabel='Newton decrement', title='Ordinary Armijo traces')
    axes[0].legend(fontsize=7)
    axes[1].semilogy(analytic.iteration, np.maximum(analytic.decrement, 1e-16), 'o-', color=COLORS['barrier'])
    axes[1].axhline(.25, color='black', linestyle=':')
    axes[1].set(xlabel='Newton direction', ylabel='Newton decrement', title='SVM: analytic damping, then full Newton steps')
    save(fig, output, '07_self_concordance')
