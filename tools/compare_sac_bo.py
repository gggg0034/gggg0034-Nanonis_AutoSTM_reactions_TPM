"""compare_sac_bo.py -- SAC and Bayesian optimisation on the same budget.

DIAGNOSTIC, NOT THE ANALYSIS BEHIND SUPPLEMENTARY FIG. 15.
    This script drives the reduced single-molecule simulator in this repository
    and reports what it measures.  It is not the five-seed analysis reported in
    Supplementary Note 6, the numbers it prints are single-seed, and they will
    not reproduce that note digit for digit.  It is provided so that both
    methods can be exercised on the same environment in a few minutes.

Both methods drive the same environment (env.py), with the same four-parameter
action space and the same reward, and both are given the same number of pulses.

Panels
    (a) running success rate against the number of pulses
    (b) cumulative mean reward against the number of pulses
    (c, d) where each method spends its pulses in the (bias, current) plane
    (e) distribution of the change in bias between consecutive decisions

Run (from the repository root):
    python tools/compare_sac_bo.py
    python tools/compare_sac_bo.py --pulses 2000 --seeds 0 1 2

Outputs (./tools/compare_output/):
    sac_pulses.csv, bo_pulses.csv   per-pulse records
    summary.json                    final-window statistics
    sac_vs_bo.png                   the figure
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

# make the repository root importable when this file is run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_output')


def _running(values, window):
    """Running mean of `values` over a trailing window of `window` samples."""
    out = np.empty(len(values), dtype=float)
    cumulative = np.cumsum(np.insert(np.asarray(values, dtype=float), 0, 0.0))
    for i in range(len(values)):
        lo = max(0, i + 1 - window)
        out[i] = (cumulative[i + 1] - cumulative[lo]) / (i + 1 - lo)
    return out


def run_sac(pulses, seed=0):
    """Run SAC for `pulses` pulses and return the per-pulse record.

    The loop mirrors SACAgent.train(): select an action, step the environment,
    add the transition, update when the buffer is warm enough, reset on `done`.
    Recording per-pulse outcomes is the only addition.
    """
    import torch
    from env import Env
    from SAC import SACAgent

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = Env(polar_space=False)
    agent = SACAgent(env)

    state = agent.one_hot_encode(env.reset())
    records = []
    for _ in range(pulses):
        action, _ = agent.select_action(state, detach_action=True)
        next_state, reward, done, info = env.step(action)
        next_state = agent.one_hot_encode(next_state)
        agent.replay_buffer.add(state, action, reward, next_state, done, info)
        if agent.replay_buffer.size() > agent.batch_size:
            agent.update()
        records.append({
            'reward': float(reward),
            'reaction': info.get('reaction', 'none'),
            'v': float(action[2]),
            'i': float(action[3]),
            'x': float(action[0]),
            'y': float(action[1]),
        })
        state = agent.one_hot_encode(env.reset()) if done else next_state
    return records


def run_bo(pulses, n_repeats=10, seed=0):
    """Run the BO baseline and flatten its history into one record per pulse."""
    from env import Env
    from BO import BOAgent

    n_repeats = max(1, int(n_repeats))
    evaluations = max(1, int(round(pulses / float(n_repeats))))
    n_init = max(1, int(round(0.06 * evaluations)))
    n_iter = max(1, evaluations - n_init)

    env = Env(polar_space=False)
    agent = BOAgent(env, n_init=n_init, n_iter=n_iter, n_repeats=n_repeats,
                    objective='reward', xi=0.02, n_candidates=1024,
                    refit_every=10, max_train_points=None, patience=None,
                    progress=True, seed=seed)
    agent.train(visualize=False)

    records = []
    for result in agent.history:
        for trial in result.get('trials', []):
            records.append({
                'reward': float(trial['reward']),
                'reaction': trial['reaction'],
                'v': float(result['action'][2]),
                'i': float(result['action'][3]),
                'x': float(result['action'][0]),
                'y': float(result['action'][1]),
            })
    return records


def _window_stats(records, fraction=0.2):
    tail = records[max(int(len(records) * (1 - fraction)), 0):]
    n = max(len(tail), 1)
    rewards = [r['reward'] for r in tail]
    deltas = _decision_deltas(records)
    return {
        'pulses': len(records),
        'tail_mean_reward': float(np.mean(rewards)) if rewards else 0.0,
        'tail_success': sum(1 for r in tail if r['reaction'] == 'success') / n,
        'tail_none': sum(1 for r in tail if r['reaction'] == 'none') / n,
        'tail_failure': sum(1 for r in tail if r['reaction'] == 'failure') / n,
        'median_delta_bias': float(np.median(deltas['dv'])) if deltas['dv'] else 0.0,
        'median_delta_current': float(np.median(deltas['di'])) if deltas['di'] else 0.0,
        'median_delta_xy': float(np.median(deltas['dxy'])) if deltas['dxy'] else 0.0,
    }


def _decision_deltas(records):
    """Change in the requested parameters between consecutive decisions.

    Consecutive *pulses* are the right unit for SAC (each pulse is a new
    decision).  For BO the decision is the parameter point, and the pulses that
    repeat it are not new decisions, so repeated identical points contribute a
    zero change and are dropped.
    """
    dv, di, dxy = [], [], []
    previous = None
    for record in records:
        current = (record['v'], record['i'], record.get('x', 0.0), record.get('y', 0.0))
        if previous is not None:
            step = abs(current[0] - previous[0])
            if step > 1e-12:                 # skip the repeats of the same BO point
                dv.append(step)
                di.append(abs(current[1] - previous[1]) * 1000.0)
                dxy.append(np.hypot(current[2] - previous[2], current[3] - previous[3]))
        previous = current
    return {'dv': dv, 'di': di, 'dxy': dxy}


def write_csv(path, records):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pulse', 'reward', 'reaction', 'bias_V', 'current_nA'])
        for index, record in enumerate(records, start=1):
            writer.writerow([index, '%.6f' % record['reward'], record['reaction'],
                             '%.6f' % record['v'], '%.6f' % record['i']])


def plot(sac_runs, bo_runs, path, window=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.0, 7.8))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(grid[0, 0:3])
    ax2 = fig.add_subplot(grid[0, 3:6])
    ax3 = fig.add_subplot(grid[1, 0:2])
    ax4 = fig.add_subplot(grid[1, 2:4])
    ax5 = fig.add_subplot(grid[1, 4:6])

    colours = {'SAC': '#1f77b4', 'BO': '#d62728'}
    for label, runs in (('SAC', sac_runs), ('BO', bo_runs)):
        for run_index, records in enumerate(runs):
            success = _running([1.0 if r['reaction'] == 'success' else 0.0 for r in records], window)
            reward = np.cumsum([r['reward'] for r in records]) / np.arange(1, len(records) + 1)
            pulses = np.arange(1, len(records) + 1)
            name = label if run_index == 0 else None
            ax1.plot(pulses, 100 * success, color=colours[label], lw=1.2,
                     alpha=0.9, label=name)
            ax2.plot(pulses, reward, color=colours[label], lw=1.2, alpha=0.9, label=name)

    ax1.set_xlabel('pulse')
    ax1.set_ylabel('success rate (%%), window = %d' % window)
    ax1.set_title('a   running success rate')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2.set_xlabel('pulse')
    ax2.set_ylabel('cumulative mean reward')
    ax2.set_title('b   cumulative mean reward')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    for ax, label, runs in ((ax3, 'SAC', sac_runs), (ax4, 'BO', bo_runs)):
        records = runs[0]
        v = np.array([r['v'] for r in records])
        i = np.array([r['i'] for r in records]) * 1000.0
        reaction = np.array([r['reaction'] for r in records])
        for name, colour, marker in (('success', '#2ca02c', 'o'),
                                     ('none', '#7f7f7f', '.'),
                                     ('failure', '#d62728', 'x')):
            mask = reaction == name
            if mask.any():
                ax.scatter(v[mask], i[mask], s=18, c=colour, marker=marker,
                           alpha=0.75, label=name)
        ax.set_xlabel('bias (V)')
        ax.set_ylabel('current (pA)')
        ax.set_title('%s   parameter cloud (%s)' % ('c' if label == 'SAC' else 'd', label))
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

    # change of the requested bias between consecutive decisions
    for label, runs, colour in (('SAC', sac_runs, colours['SAC']),
                                ('BO', bo_runs, colours['BO'])):
        merged = []
        for records in runs:
            merged.extend(_decision_deltas(records)['dv'])
        if merged:
            ax5.hist(merged, bins=np.linspace(0, 1.5, 31), alpha=0.55,
                     color=colour, label='%s (median %.3f V)' % (label, np.median(merged)))
    ax5.set_xlabel('| change in bias between decisions | (V)')
    ax5.set_ylabel('count')
    ax5.set_title('e   how far the parameters move each time')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(alpha=0.3)

    fig.suptitle('SAC policy vs Bayesian optimisation, identical environment and budget', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare SAC and BO on the same pulse budget.')
    parser.add_argument('--pulses', type=int, default=1000, help='pulses per method')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--bo-repeats', type=int, default=10,
                        help='pulses averaged per BO parameter point')
    parser.add_argument('--window', type=int, default=100,
                        help='window of the running success rate')
    args = parser.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 78)
    print('  SAC vs Bayesian optimisation, identical environment and budget')
    print('=' * 78)
    print('  pulses per method : %d' % args.pulses)
    print('  seeds             : %s' % ', '.join(str(s) for s in args.seeds))
    print('-' * 78)

    sac_runs, bo_runs, summary = [], [], {'pulses': args.pulses, 'seeds': args.seeds,
                                          'sac': [], 'bo': []}
    for seed in args.seeds:
        started = time.time()
        print('  seed %d  SAC ...' % seed, end='', flush=True)
        sac = run_sac(args.pulses, seed)
        print(' %.1f s' % (time.time() - started), flush=True)

        started = time.time()
        print('  seed %d  BO  ...' % seed, end='', flush=True)
        bo = run_bo(args.pulses, args.bo_repeats, seed)
        print(' %.1f s' % (time.time() - started), flush=True)

        sac_runs.append(sac)
        bo_runs.append(bo)
        summary['sac'].append(_window_stats(sac))
        summary['bo'].append(_window_stats(bo))

        if len(args.seeds) == 1:
            write_csv(os.path.join(OUT_DIR, 'sac_pulses.csv'), sac)
            write_csv(os.path.join(OUT_DIR, 'bo_pulses.csv'), bo)

    figure_path = os.path.join(OUT_DIR, 'sac_vs_bo.png')
    plot(sac_runs, bo_runs, figure_path, args.window)
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print('-' * 78)
    for label in ('sac', 'bo'):
        rows = summary[label]
        mean_success = 100 * np.mean([r['tail_success'] for r in rows])
        mean_reward = np.mean([r['tail_mean_reward'] for r in rows])
        median_dv = np.mean([r['median_delta_bias'] for r in rows])
        median_dxy = np.mean([r['median_delta_xy'] for r in rows])
        print('  %-4s  last 20%%: success %5.1f%%   mean reward %6.2f   '
              'median |dV| %.3f V   median |d(x,y)| %.1f px'
              % (label.upper(), mean_success, mean_reward, median_dv, median_dxy))
    print('-' * 78)
    print('  figure : %s' % figure_path)
    print('  data   : %s/' % os.path.abspath(OUT_DIR))
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
