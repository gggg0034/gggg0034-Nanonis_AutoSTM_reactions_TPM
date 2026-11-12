# AI-driven single-molecule lithography on TPM lattices

Code for the manuscript *"AI-driven single-molecule lithography towards programmable
molecular lattices"*.

The repository contains **two kinds of code that are run in different places**:

* **simulation code** — trains the agent, and runs the Bayesian-optimisation baseline, in a
  virtual tip-operation environment. Runs on any laptop, no instrument, no data.
* **instrument code** — talks to the Nanonis controller, scans, finds molecules, pulses the
  tip. Needs the microscope, the Nanonis software and a sample.

Section 2 lists which file belongs to which. If you only want to see the code work, read
section 3 and ignore the rest.

---

## 1. Install

```bash
git clone https://github.com/gggg0034/Nanonis_AutoSTM_reactions_TPM.git
cd Nanonis_AutoSTM_reactions_TPM

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

Tested with Python 3.8, PyTorch 1.8.2, gym 0.26.2, OpenCV 4.6, NumPy 1.21, on Windows.
`gym` must stay at 0.26.x — the code imports `gym.spaces`, which `gymnasium` renamed.

---

## 2. Simulation or instrument: which file is which

| File | What it does | Instrument needed | How to run |
|---|---|---|---|
| `env.py` | the virtual tip-operation environment: molecule, tip, pulse, stochastic outcome | **no** | imported by `SAC.py` and `BO.py` |
| `SAC.py` | the SAC agent **and** its training loop, acting in `env.py` | **no** | `python SAC.py` |
| `BO.py` | the Gaussian-process Bayesian-optimisation baseline, acting in `env.py` | **no** | `python BO.py` |
| `demo_pipeline.py` | one pass of the workflow on a recorded scan: image → keypoints → parameters → outcome | **no** | `python demo_pipeline.py` |
| `train_sim.py` | wrapper around `SAC.py`: quiet log, CLI options, end-of-run summary | **no** | `python train_sim.py` |
| `run_bo.py` | wrapper around `BO.py`: configuration of Supplementary Table 4, CLI options, summary | **no** | `python run_bo.py` |
| `tools/compare_sac_bo.py` | both methods, same budget, one figure (diagnostic) | **no** | `python tools/compare_sac_bo.py` |
| `AI_Tip_TPM_Main.py` | the autonomous experiment: scan, recognise, decide, pulse | **yes** | `python AI_Tip_TPM_Main.py --real` |
| `AI_Tip_TPM_Main_fixed.py` | same, with a fixed pattern and no agent | **yes** | `python AI_Tip_TPM_Main_fixed.py --real` |
| `Auto_scan_class.py` | scanning, tip approach, pulsing, image handling — the instrument class | **yes** | used by the two entry points above |
| `core.py` | instrument controller on top of the TCP protocol | **yes** | " |
| `interface.py` | the Nanonis TCP protocol | **yes** | " |
| `SAC_H_nanonis.py` | the same SAC networks as `SAC.py`, but coupled to the running instrument through queues | **yes** | " |
| `env_TPM.py` | multi-site state and parameter grid used on the instrument side | **yes** | " |
| `molecule_registry.py`, `mol_vis.py` | molecule bookkeeping, drift correction, keypoint heatmaps | **yes** | " |
| `utils.py` | image, geometry and fitting helpers | shared | used by both halves |

Rule of thumb: **`AI_Tip_TPM_*.py` and the instrument class behind it need the microscope;
`SAC.py`, `BO.py`, `env.py` and `demo_pipeline.py` do not.**

`SAC.py` and `SAC_H_nanonis.py` are the same algorithm. `SAC.py` draws its transitions from
`env.py`, so it can train with no hardware. `SAC_H_nanonis.py` receives them from the
instrument and is what actually turns the tip during a real run.

---

## 3. Simulation — runs anywhere

### 3.1 Train the SAC agent — `SAC.py`

```bash
python SAC.py
```

`SAC.py` holds the agent and its training loop. Running it trains for 1,000 pulses in the
virtual environment and writes

* `models/<timestamp>/policy_net_<step>.pth`, `q_net1_*`, `q_net2_*` — the trained networks;
* `train_result_plot/action_<...>.png` — action distribution and return;
* `train_result_plot/buffer_data_<...>.txt` — every transition.

Nothing leaves the process: the "tip" is a coordinate in `env.py` and the "reaction" is
drawn from the model in `env.py`. The reward is

| outcome | reward |
|---|---|
| conversion | **+10** |
| no reaction | **−1** |
| degradation | **−5** |

`SAC.py` prints one line per pulse, which is noisy over a long run. For a quiet log with a
summary at the end:

```bash
python train_sim.py                 # 1,000 pulses, about 20 s on a CPU
python train_sim.py --steps 3000
```

At the end of 1,000 pulses the agent typically sits at **≈60 % conversion, ≈10 %
degradation, ≈30 % no reaction over the last 200 pulses**, the regime reported for the
training stage of the paper.

### 3.2 Run the Bayesian-optimisation baseline — `BO.py`

```bash
python BO.py
```

`BO.py` is the baseline of Supplementary Note 6, acting in the same `env.py` with the same
reward and the same action bounds. It runs Gaussian-process regression with a Matérn 5/2
kernel and automatic relevance determination, and selects points by expected improvement.
Running the file uses the settings in its `__main__` block and writes the optimisation
history, the per-pulse record, the best parameter set and a figure to `bo_result_plot/`.

For the configuration of Supplementary Table 4 (ξ = 0.01, 3,000 candidates, hyperparameters
refitted every 50 observations, 100 evaluations × 10 repeats):

```bash
python run_bo.py                    # about 4 s
```

The optimum it reports is typically within about 0.1 V of −4.0 V and 0.05 nA of 0.22 nA, at
84–96 % conversion.

> `tools/compare_sac_bo.py` runs both methods with the same pulse budget and puts them on
> one figure. It is a diagnostic on this reduced simulator, not the five-seed analysis of
> Supplementary Fig. 15, and it will not reproduce that note digit for digit.

### 3.3 Replay one pass of the workflow — `demo_pipeline.py`

```bash
python demo_pipeline.py
```

The same four stages as an experiment, on recorded data:

1. **acquisition** — the scan is read from `assets/demo_scan.png` instead of from the
   instrument (304 × 304 px, the size of the live scan buffer).
2. **recognition** — the image goes to the YOLOv7 keypoint detector. The detector weights
   are not distributed with this repository (see §7), so when they are absent the recorded
   keypoint file `assets/demo_keypoints.txt` is used instead and the keypoints are drawn on
   the image. **A missing detector is not an error.**
3. **decision** — the SAC policy proposes (x, y, bias, current). With no trained checkpoint
   the policy returns the nominal operating point of the paper, **−4.0 V / 100 pA**, instead
   of failing.
4. **outcome** — the parameters are applied to the simulator and the reaction is reported.

Outputs land in `demo_output/`: `01_scan.png`, `02_keypoints.png` (keypoints drawn on the
lattice), `03_summary.png`. A typical run is 40 pulses, of which roughly two thirds convert,
in a few seconds.

---

## 4. Real experiment — needs the microscope

Only this section touches hardware. Nothing in §3 does.

```bash
python AI_Tip_TPM_Main.py --real                # autonomous, SAC agent
python AI_Tip_TPM_Main_fixed.py --real          # fixed pattern, no agent
```

Without `--real` both entry points run the recorded-data demo of §3.3 instead.

**Prerequisites**

* A prepared sample: the TPM precursor deposited on the surface, and a molecular island
  found in the scan area.
* A tip that has been approached and is stable.
* Nanonis running, with a license that **includes the TCP/IP programming interface**.
  Without that license the server socket never opens, so the program cannot connect even
  though Nanonis is running.
* The programming interface enabled in Nanonis
  (*File → Preferences → TCP Programming Interface*) on port 6501. Nanonis accepts one
  client per port; if 6501 is taken, pass `--port 6502` (or 6503, 6504).

With those in place the program starts the autonomous run as deposited.

If the port is closed the program prints a checklist — including the license point above —
and exits, rather than dying with a `ConnectionRefusedError` inside the import chain:

```
==============================================================================
  Cannot connect to the Nanonis controller.
------------------------------------------------------------------------------
  What this means
    The program talks to Nanonis through a TCP socket on 127.0.0.1:6501.
  ...
```

`config.yaml` is read **only** by `AI_Tip_TPM_Main_fixed.py`; `AI_Tip_TPM_Main.py` keeps
its constants in the source next to the loop that uses them.

---

## 5. Repository layout

```
.
│  ── simulation ─────────────────────────────────────────────────────────────
├── env.py                       virtual tip-operation environment
├── SAC.py                       SAC agent + training loop
├── BO.py                        Bayesian-optimisation baseline
├── train_sim.py                 wrapper: quiet SAC training with a summary
├── run_bo.py                    wrapper: BO with the Supplementary Table 4 settings
├── demo_pipeline.py             recorded-scan demo of the whole data path
├── tools/compare_sac_bo.py      diagnostic: both methods, one figure
│
│  ── instrument ─────────────────────────────────────────────────────────────
├── AI_Tip_TPM_Main.py           autonomous experiment, SAC agent
├── AI_Tip_TPM_Main_fixed.py     fixed-pattern variant, no agent
├── Auto_scan_class.py           Mustard_AI_Nanonis: scanning, approach, pulsing
├── core.py                      instrument controller, above interface.py
├── interface.py                 Nanonis TCP protocol (CoccaGuo, extended)
├── SAC_H_nanonis.py             SAC networks coupled to the live instrument
├── env_TPM.py                   multi-site state/grid used on the instrument side
├── molecule_registry.py         molecule bookkeeping and drift correction
├── mol_vis.py                   keypoint heatmap visualisation
├── nanonis_guard.py             TCP probe and the connection-error message
│
│  ── shared ─────────────────────────────────────────────────────────────────
├── utils.py                     image, geometry and fitting helpers
├── config.yaml                  settings for AI_Tip_TPM_Main_fixed.py
├── requirements.txt
├── LICENSE
├── assets/
│   ├── demo_scan.png            recorded 304 x 304 scan used by the demo
│   └── demo_keypoints.txt       keypoints for that scan
└── README.md
```

`assets/demo_keypoints.txt` is the detector output for the recorded scan, one row per
molecule: `class  x  y  w  h  key_x  key_y`, coordinates normalised to [0, 1]. It is the
same format the detector returns, so the fallback and the detector are interchangeable
downstream.

---

## 6. Which script corresponds to which part of the paper

| Paper | Script |
|---|---|
| Supplementary Note 5, environment and reward | `env.py` |
| Supplementary Table 1, SAC hyperparameters | `SAC.py` |
| Supplementary Note 6 / Table 4, BO configuration | `BO.py`, `run_bo.py` |
| autonomous fabrication loop, Fig. 2 and Fig. 3 | `AI_Tip_TPM_Main.py` |
| recorded-keypoint fallback, Fig. 1 and Supplementary Fig. 2 | `demo_pipeline.py`, `assets/demo_keypoints.txt` |

---

## 7. What is **not** included, and why

The following belong to the computer-vision layer, and their weights are not public:

| Path | What it is | Consequence |
|---|---|---|
| `keypoint/detect.py`, `keypoint/best_0723.pt` | YOLOv7 keypoint detector | absent → `demo_pipeline.py` falls back to the recorded keypoint file; `Auto_scan_class.py` raises one clear message if a live scan needs the detector |
| `EvaluationCNN/`, `mol_segment/` | scan-quality CNN and molecule segmentation U-Net | only used by the live acquisition path |
| `DQN/` | an earlier tip-shaping agent, superseded by SAC | not imported anywhere in this version |

None of these is needed for the results reported in the paper, and none is needed to run
any command in §3. They are available from the authors on request.

---

## 8. Notes and limitations of the simulator

* `env.py` reduces the surface to a **single molecule**. The lattice geometry and the
  tip-position dependence of the reaction probability are kept, but intermolecular effects
  and the full multi-site state machine are not simulated. `env_TPM.py` is the multi-site
  environment that runs alongside the instrument; it is **not** the environment behind
  Supplementary Note 6 and is not intended for standalone training.
* The SAC agent samples minibatches uniformly from the replay buffer.
* The entropy temperature is **fixed at 0.2** in `SAC.py` and `SAC_H_nanonis.py`, as listed
  in Supplementary Table 1. It is not adapted during training.
* A single run of `SAC.py` / `train_sim.py` is one seed, and the default budget of 1,000
  pulses matches the training budget of the paper (1,040 pulses). The hyperparameters are
  tuned for that budget; they are not tuned for much longer runs.

---

## 9. Citation and license

If you use this code, please cite the manuscript. Released under the license in `LICENSE`.
The TCP protocol implementation in `interface.py` follows CoccaGuo's
`nanonis-programming-interface` and is redistributed under the same terms.
