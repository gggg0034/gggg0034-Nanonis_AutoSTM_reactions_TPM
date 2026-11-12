"""train_sim.py -- train the SAC agent in the virtual tip-operation environment.

This is the training half of the workflow, with no instrument involved.  The
environment is env.py: a triangular TPM molecule on a lattice, a tip that can be
placed anywhere relative to the molecule, and a pulse described by
(bias, current).  The reaction outcome is stochastic with a success probability
that decays with the distance between the tip and the molecule and with the
deviation of the parameters from the optimum; the reward is

    success  +10        no reaction  -1        degraded  -5

which is the reward used throughout the paper.

Run:
    python train_sim.py                 # 2000 pulses
    python train_sim.py --steps 5000
    python train_sim.py --polar-space   # actions in (length, angle) instead of (x, y)

Outputs:
    ./models/<timestamp>/policy_net_<step>.pth     policy checkpoint
    ./train_result_plot/action_<...>.png           action distribution + return
"""

import argparse
import os
import random
import sys
import time

import numpy as np


class _DropRewardLines:
    """Silence the per-step 'Reward: ...' print so the log stays readable."""

    def __init__(self, stream):
        self.stream = stream
        self._skip_line = False

    def write(self, text):
        kept = []
        for chunk in text.splitlines(True):
            if self._skip_line:
                if chunk.endswith('\n'):
                    self._skip_line = False
                continue
            if chunk.startswith('Reward:'):
                if not chunk.endswith('\n'):
                    self._skip_line = True
                continue
            kept.append(chunk)
        if kept:
            self.stream.write(''.join(kept))
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def summarise(buffer, tail_fraction=0.2):
    """Outcome counts and mean reward over the whole buffer and over its tail."""
    items = list(buffer)
    if not items:
        return {}
    def stats(rows):
        reactions = {'success': 0, 'none': 0, 'failure': 0}
        rewards = []
        for row in rows:
            info = row[5] if len(row) > 5 and isinstance(row[5], dict) else {}
            reactions[info.get('reaction', 'none')] = reactions.get(info.get('reaction', 'none'), 0) + 1
            rewards.append(float(row[2]))
        n = max(len(rows), 1)
        return {
            'n': len(rows),
            'mean_reward': float(np.mean(rewards)),
            'success': reactions['success'] / n,
            'none': reactions['none'] / n,
            'failure': reactions['failure'] / n,
        }
    cut = max(int(len(items) * (1.0 - tail_fraction)), 0)
    return {'all': stats(items), 'last': stats(items[cut:])}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train SAC in the TPM simulator (no instrument).')
    parser.add_argument('--steps', type=int, default=1000, help='number of pulses (default 1000, the budget used in the paper)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--polar-space', action='store_true',
                        help='use (length, angle) actions instead of (x, y)')
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        raise SystemExit('PyTorch is required:  pip install -r requirements.txt')

    from env import Env
    from SAC import SACAgent

    env = Env(polar_space=args.polar_space)
    agent = SACAgent(env)

    print('=' * 78)
    print('  SAC training in the virtual tip-operation environment')
    print('=' * 78)
    print('  device      : %s' % agent.device)
    print('  pulses      : %d' % args.steps)
    print('  action space: %s .. %s' % (np.round(env.action_space.low, 3),
                                        np.round(env.action_space.high, 3)))
    print('  reward      : success +10 | no reaction -1 | degraded -5')
    print('-' * 78)

    started = time.time()
    original_stdout = sys.stdout
    sys.stdout = _DropRewardLines(original_stdout)
    try:
        agent.train(args.steps, visualize=False)
    finally:
        sys.stdout = original_stdout
    elapsed = time.time() - started

    report = summarise(agent.replay_buffer.buffer)
    print('-' * 78)
    print('  finished in %.1f s' % elapsed)
    for label, key in (('all pulses', 'all'), ('last 20%', 'last')):
        block = report.get(key)
        if not block:
            continue
        print('  %-10s n=%4d  mean reward %6.2f   success %5.1f%%   no reaction %5.1f%%   degraded %5.1f%%'
              % (label, block['n'], block['mean_reward'],
                 100 * block['success'], 100 * block['none'], 100 * block['failure']))
    print('-' * 78)
    print('  checkpoint : ./models/')
    print('  figure     : ./train_result_plot/')
    print('  next       : python run_bo.py     (Bayesian-optimisation baseline)')
    print('               python tools/compare_sac_bo.py  (both, on one figure)')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
