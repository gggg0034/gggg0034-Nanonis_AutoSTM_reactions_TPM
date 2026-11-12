"""demo_pipeline.py -- run the workflow without an instrument.

A reviewer who does not have a Nanonis controller can still follow the three
modules that make up one pulse of the workflow, in the same order and on the
same code that an experiment uses:

    1. acquisition   a scan is read.  Demo mode replays a recorded image
                     (assets/demo_scan.png) instead of asking the instrument.
    2. recognition   the image is passed to the keypoint detector.  When the
                     detector or its weights are not available the recorded
                     keypoint file (assets/demo_keypoints.txt) is used instead,
                     and the keypoints are drawn on the image so the output can
                     be inspected.
    3. decision      the SAC policy proposes (x, y, V, I).  When no trained
                     checkpoint is available the policy returns the nominal
                     operating point used in the paper, -4.0 V / 100 pA, rather
                     than raising an error.
    4. outcome       the parameters are applied to the molecular simulator and
                     the reaction outcome is reported.

Nothing here is a substitute for the experiment; it exists so that the data
path can be exercised and inspected on a laptop.

Usage:
    python demo_pipeline.py
    python demo_pipeline.py --out demo_output --limit 20
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'assets')
DEMO_SCAN = os.path.join(ASSETS, 'demo_scan.png')
DEMO_KEYPOINTS = os.path.join(ASSETS, 'demo_keypoints.txt')

# Nominal operating point reported in the paper, used when the policy has not
# been trained (no checkpoint).  action = [x, y, V, I]; x and y are offsets
# from the molecule centre, so 0, 0 aims at the centre of the site.
NOMINAL_ACTION = np.array([0.0, 0.0, -4.0, 0.1], dtype=np.float32)


def _banner(text):
    print()
    print('=' * 78)
    print('  ' + text)
    print('=' * 78)


def _step(text):
    print('\n[%s] %s' % (_now(), text))


def _now():
    import time
    return time.strftime('%H:%M:%S')


def _imread_gray(path):
    """Read a grayscale image.  cv2.imread silently fails on non-ASCII paths."""
    import cv2
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _imwrite(path, image):
    """Write an image.  cv2.imwrite silently fails on non-ASCII paths."""
    import cv2
    extension = os.path.splitext(path)[1] or '.png'
    ok, buffer = cv2.imencode(extension, image)
    if ok:
        buffer.tofile(path)
    return ok


# ---------------------------------------------------------------------------
# 1. acquisition
# ---------------------------------------------------------------------------
def load_recorded_scan(size=304):
    """Return the recorded scan as a uint8 grayscale image, resized to `size`."""
    import cv2
    if not os.path.exists(DEMO_SCAN):
        raise SystemExit('Recorded scan not found: %s' % DEMO_SCAN)
    image = _imread_gray(DEMO_SCAN)
    if image is None:
        raise SystemExit('Could not read the recorded scan: %s' % DEMO_SCAN)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# 2. recognition
# ---------------------------------------------------------------------------
def read_keypoint_file(path=DEMO_KEYPOINTS):
    """Read a keypoint file: one row per molecule.

    Columns: class  x  y  w  h  key_x  key_y   (all coordinates normalised to
    [0, 1] except the class label).  This is the format written by the
    detector, so the fallback and the detector are interchangeable downstream.
    """
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append(tuple(float(value) for value in parts[:7]))
    return rows


def _yolo_keypoints(image, model_path):
    """Run the YOLOv7 keypoint detector if it is installed; return None otherwise."""
    if not os.path.exists(model_path):
        return None
    try:
        from keypoint.detect import key_detect       # optional dependency
    except Exception as exc:
        print('  YOLO keypoint module unavailable (%s: %s)' % (type(exc).__name__, exc))
        return None
    try:
        return key_detect(image, model_path, os.path.join(HERE, 'demo_output'))
    except Exception as exc:
        print('  keypoint detector failed (%s: %s)' % (type(exc).__name__, exc))
        return None


def detect_keypoints(image, model_path='./keypoint/best_0723.pt'):
    """Detector first, recorded keypoints second.  Never raises for a missing model."""
    result = _yolo_keypoints(image, model_path)
    if result:
        print('  keypoints: %d from the YOLO keypoint detector' % len(result))
        return result, 'yolo'
    print('  no detector weights at %s' % model_path)
    print('  -> falling back to the recorded keypoint file (no error)')
    rows = read_keypoint_file()
    print('  keypoints: %d from %s' % (len(rows), os.path.basename(DEMO_KEYPOINTS)))
    return rows, 'recorded'


def draw_keypoints(image, keypoints, path):
    """Draw boxes and keypoints on a copy of the image and save it."""
    import cv2
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    for row in keypoints:
        _, cx, cy, w, h = row[0], row[1], row[2], row[3], row[4]
        x1 = int((cx - w / 2.0) * width)
        y1 = int((cy - h / 2.0) * height)
        x2 = int((cx + w / 2.0) * width)
        y2 = int((cy + h / 2.0) * height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 1)
        for index in range(5, len(row), 2):
            kx, ky = int(row[index] * width), int(row[index + 1] * height)
            cv2.drawMarker(canvas, (kx, ky), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
    _imwrite(path, canvas)
    return canvas


# ---------------------------------------------------------------------------
# 3. decision
# ---------------------------------------------------------------------------
def make_decision_maker(checkpoint_dir='./SAC/train_results'):
    """Return (function, description).

    With a trained checkpoint the SAC policy is used.  Without one the function
    returns the nominal operating point and says so; it never raises.
    """
    if not os.path.isdir(checkpoint_dir):
        return (lambda state: NOMINAL_ACTION.copy(),
                'fixed nominal parameters (-4.0 V, 100 pA) -- no SAC checkpoint')

    try:
        from SAC_H_nanonis import SACAgent, Env
    except Exception as exc:
        return (lambda state: NOMINAL_ACTION.copy(),
                'fixed nominal parameters -- SAC stack unavailable (%s)' % type(exc).__name__)

    try:
        env = Env(polar_space=False)
        agent = SACAgent(env)
        try:
            agent.load_model(checkpoint_dir)
        except Exception as exc:
            return (lambda state: NOMINAL_ACTION.copy(),
                    'fixed nominal parameters -- no usable checkpoint in %s (%s)'
                    % (checkpoint_dir, type(exc).__name__))

        def policy(state):
            action, _ = agent.select_action(np.asarray([state], dtype=np.float32),
                                            detach_action=True)
            return np.asarray(action, dtype=np.float32).ravel()[:4]

        return policy, 'trained SAC policy from %s' % checkpoint_dir
    except Exception as exc:
        return (lambda state: NOMINAL_ACTION.copy(),
                'fixed nominal parameters -- SAC unavailable (%s)' % type(exc).__name__)


# ---------------------------------------------------------------------------
# 4. outcome
# ---------------------------------------------------------------------------
def simulate_outcome(action):
    """Apply one action to the molecular simulator and report the outcome."""
    from env import Env
    env = Env(polar_space=False)
    env.reset()
    _, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
    return reward, info


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(out_dir='demo_output', limit=None, seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    _banner('DEMO MODE - recorded data, no instrument')
    print('  Nothing is sent to a Nanonis controller. The scan is replayed from')
    print('  assets/demo_scan.png and the keypoints from assets/demo_keypoints.txt.')
    print('  For a real experiment run:  python AI_Tip_TPM_Main.py --real')

    _step('1/4  acquisition')
    image = load_recorded_scan()
    scan_path = os.path.join(out_dir, '01_scan.png')
    import cv2
    _imwrite(scan_path, image)
    print('  scan: %dx%d px  ->  %s' % (image.shape[1], image.shape[0], scan_path))

    _step('2/4  recognition')
    keypoints, source = detect_keypoints(image)
    if limit:
        keypoints = keypoints[:limit]
    kp_path = os.path.join(out_dir, '02_keypoints.png')
    draw_keypoints(image, keypoints, kp_path)
    print('  annotated image -> %s  (source: %s)' % (kp_path, source))

    _step('3/4  decision')
    policy, description = make_decision_maker()
    print('  policy: ' + description)
    actions = []
    for row in keypoints:
        actions.append(np.asarray(policy(row[0]), dtype=np.float32))
    unique = np.unique(np.round(np.array(actions), 3), axis=0)
    print('  proposed (V, I): %s' % ', '.join(
        '(%.2f V, %.0f pA)' % (a[2], a[3] * 1000) for a in unique[:5]))

    _step('4/4  outcome on the simulator')
    reactions = {'success': 0, 'none': 0, 'failure': 0}
    rewards = []
    for action in actions:
        reward, info = simulate_outcome(action)
        reactions[info.get('reaction', 'none')] = reactions.get(info.get('reaction', 'none'), 0) + 1
        rewards.append(reward)
    total = max(len(rewards), 1)
    print('  pulses: %d   mean reward: %.2f' % (total, float(np.mean(rewards)) if rewards else 0.0))
    for name in ('success', 'none', 'failure'):
        print('    %-8s %3d  (%5.1f%%)' % (name, reactions[name], 100.0 * reactions[name] / total))

    _plot_summary(out_dir, actions, rewards, reactions)
    _banner('DEMO FINISHED')
    print('  outputs in: %s' % os.path.abspath(out_dir))
    print('  next:  python train_sim.py     (train the SAC agent)')
    print('         python run_bo.py        (Bayesian-optimisation baseline)')
    print()
    return 0


def _plot_summary(out_dir, actions, rewards, reactions):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    if len(actions):
        arr = np.array(actions)
        axes[0].scatter(arr[:, 2], arr[:, 3] * 1000, s=22)
        axes[0].set_xlabel('bias (V)')
        axes[0].set_ylabel('current (pA)')
        axes[0].set_title('proposed parameters')
        axes[0].grid(alpha=0.3)
    labels = [key for key in ('success', 'none', 'failure')]
    axes[1].bar(labels, [reactions[key] for key in labels],
                color=['#2ca02c', '#7f7f7f', '#d62728'])
    axes[1].set_ylabel('pulses')
    axes[1].set_title('outcome on the simulator')
    fig.tight_layout()
    path = os.path.join(out_dir, '03_summary.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('  summary figure -> %s' % path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the workflow on recorded data (no instrument).')
    parser.add_argument('--out', default='demo_output', help='output directory')
    parser.add_argument('--limit', type=int, default=None, help='use only the first N keypoints')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    return run(args.out, args.limit, args.seed)


if __name__ == '__main__':
    sys.exit(main())
