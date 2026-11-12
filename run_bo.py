"""run_bo.py -- Bayesian-optimisation baseline in the same virtual environment.

The baseline searches the same four parameters (x, y, bias, current) that the
SAC agent controls, in the same environment, with the same reward, so the two
can be compared on equal terms.  It is the baseline referred to in
Supplementary Note 6 and Supplementary Fig. 15.

Method (settings from Supplementary Table 4)
    - 6 Latin-hypercube points, then 94 sequential EI steps (100 evaluations)
    - each parameter point is repeated 10 times and averaged, because a single
      pulse carries a lot of stochasticity; 100 x 10 = 1,000 tip operations
    - the surrogate is a Gaussian process with a Matern 5/2 kernel and ARD;
      hyperparameters are re-fitted every 50 observations by maximising the log
      marginal likelihood
    - the acquisition function is Expected Improvement with xi = 0.01, evaluated
      on 3,000 random candidates plus local perturbations around the current
      best, followed by a shrinking local search

The point of the comparison is not that one method is faster.  Bayesian
optimisation evaluates the objective at isolated, widely separated parameter
points; the policy moves through the parameter space and settles.  See
Supplementary Note 6.

Run:
    python run_bo.py
    python run_bo.py --n-init 6 --n-iter 94 --n-repeats 10 --seed 0

Outputs (./bo_result_plot/):
    bo_history_<time>.txt     every evaluation: parameters, reward, success rate
    bo_trials_<time>.txt      every individual pulse
    bo_best_<time>.json       best parameters found
    bo_result_<time>_<hp>.png parameter distribution and optimisation curve
"""

import argparse
import os
import sys
import time

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bayesian-optimisation baseline (no instrument).')
    parser.add_argument('--n-init', type=int, default=6, help='Latin-hypercube initial points')
    parser.add_argument('--n-iter', type=int, default=94, help='sequential EI steps')
    parser.add_argument('--n-repeats', type=int, default=10,
                        help='pulses per parameter point (averaged)')
    parser.add_argument('--objective', default='reward', choices=['reward', 'success'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quiet', action='store_true', help='no progress bar')
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    from env import Env
    from BO import BOAgent

    env = Env(polar_space=False)
    agent = BOAgent(env,
                    n_init=args.n_init,
                    n_iter=args.n_iter,
                    n_repeats=args.n_repeats,
                    objective=args.objective,
                    xi=0.01,                 # Supplementary Table 4
                    n_candidates=3000,       # Supplementary Table 4
                    refit_every=50,          # Supplementary Table 4
                    max_train_points=300,
                    patience=None,
                    progress=not args.quiet,
                    seed=args.seed)

    evaluations = args.n_init + args.n_iter
    print('=' * 78)
    print('  Bayesian optimisation in the virtual tip-operation environment')
    print('=' * 78)
    print('  evaluations : %d  (%d initial + %d sequential)'
          % (evaluations, args.n_init, args.n_iter))
    print('  pulses      : %d  (%d per parameter point)'
          % (evaluations * args.n_repeats, args.n_repeats))
    print('  objective   : %s' % args.objective)
    print('  reward      : success +10 | no reaction -1 | degraded -5')
    print('-' * 78)

    started = time.time()
    best_action, best_score = agent.train(visualize=False)
    elapsed = time.time() - started

    names = agent.action_names()
    print('-' * 78)
    print('  finished in %.1f s' % elapsed)
    print('  best parameters: ' + ', '.join(
        '%s=%.3f' % (name, value) for name, value in zip(names, best_action)))
    if agent.best_result:
        print('  best success rate: %.1f%%   mean reward: %.2f'
              % (100 * agent.best_result['success_rate'], agent.best_result['reward']))
    print('-' * 78)
    print('  results: ./bo_result_plot/')
    print('  next   : python tools/compare_sac_bo.py   (SAC and BO on one figure)')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
