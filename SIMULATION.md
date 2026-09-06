# Simulation — from a clean machine to a video, command by command

Run the controller against a simulated crowd: a real Unitree Go2 model, pedestrians that react
to the robot, walls, metrics, and an mp4. No robot, no GPU, no ROS. About five minutes.

Every command below was executed on a clean Linux box and a Windows box while writing this
guide; the printed output is real, not illustrative.

---

## Step 1 — Get the code

```bash
git clone https://github.com/elmoghany/crowd-nav
cd crowd-nav
```

## Step 2 — Make an isolated environment

Keeps this separate from anything else on your machine.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

**If that fails with `ensurepip is not available`** (common on Debian/Ubuntu, and exactly what
happened on the machine this guide was tested on) you do not need sudo -- build the venv without
pip and bootstrap it:

```bash
python3 -m venv --without-pip .venv
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
```

## Step 3 — Install the simulator

```bash
pip install -r requirements.txt          # controller: torch, numpy, huggingface_hub
pip install -r requirements-sim.txt      # simulator:  mujoco, imageio, imageio-ffmpeg, pyyaml
```

Takes 1–3 minutes, mostly PyTorch. Check it landed:

```bash
python -c "import mujoco, torch; print('mujoco', mujoco.__version__, '| torch', torch.__version__)"
```

```
mujoco 3.12.0 | torch 2.14.0+cu130
```

## Step 4 — Run your first simulation (no weights, no video)

```bash
python -m sim.simulate --scenario corridor
```

```
  scenario               corridor
  mode                   sfm
  reached                True
  sim_time_s             31.2
  min_clearance_m        0.732
  contact_steps          0
  mean_ped_deviation_m   0.03
  peak_ped_deviation_m   0.081
  plan_ms_mean           3.58
  plan_ms_p95            35.64
  pedestrians            6
  compliance             1.0

  clean run: reached the goal with no contact.
```

What those numbers mean:

| field | meaning |
|---|---|
| `reached` | did the robot get within 0.5 m of the goal before the time limit |
| `min_clearance_m` | closest surface-to-surface gap to any person all episode (negative = contact) |
| `contact_steps` | number of simulation steps spent overlapping a person |
| `mean_ped_deviation_m` | how far the crowd is from where it would have walked **with no robot present**, averaged over the episode — the robot's causal footprint, computed by running a second, robot-free copy of the same crowd. `peak_` is its maximum |
| `plan_ms_*` | wall-clock per decision. This is the real-time budget check |

## Step 5 — Record a video

(Working on a Linux server and want to watch from your laptop instead? `--serve` streams the run to a browser -- see **[REMOTE.md](REMOTE.md)**.)

```bash
python -m sim.simulate --scenario corridor --video corridor.mp4 --camera top
```

Adds `video corridor.mp4` and `frames 313` to the output. Cameras: `top` (clearest for reading
who yields), `side` (3/4 view), `follow` (rides behind the robot).

**On a headless Linux machine** (no display), tell MuJoCo to render offscreen first:

```bash
export MUJOCO_GL=egl        # if that fails: pip install PyOpenGL osmesa && export MUJOCO_GL=osmesa
python -m sim.simulate --scenario corridor --video corridor.mp4
```

## Step 6 — Run the other scenarios

```bash
python -m sim.simulate --scenario crossing --video crossing.mp4
python -m sim.simulate --scenario doorway  --video doorway.mp4 --camera top
```

`doorway` is the hard one — a 1.6 m gap that both the robot and an oncoming group must pass.

## Step 7 — Use the trained model instead of the physics baseline

Steps 4–6 ran `--mode sfm`, which needs no weights. To use the learned response model:

```bash
python -m sim.simulate --scenario doorway --mode residual --video doorway_learned.mp4
```

It downloads `residual_predictor_k1.pt` from Hugging Face the first time (~90 KB, cached
afterwards). Compare the two runs — `mode` changes only how the robot *predicts people*, not how
it plans.

## Step 8 — Make the crowd less cooperative

The crowd yields to the robot by default. `--compliance` scales how strongly people react to it:

```bash
python -m sim.simulate --scenario doorway --compliance 1.0     # cooperative (default)
python -m sim.simulate --scenario doorway --compliance 0.3     # reluctant
python -m sim.simulate --scenario doorway --compliance 0.1     # essentially ignores the robot
```

Measured on the `doorway` scenario (12 oncoming people, 1.6 m gap):

| compliance | reached | min clearance | mean ped deviation |
|---|---|---|---|
| 1.0 | yes | 0.79 m | 0.034 m |
| 0.3 | yes | 0.80 m | 0.021 m |
| 0.1 | yes | 0.78 m | 0.009 m |

Read that honestly: what changes is **how much the crowd gets out of the way** (a 4x spread in
deviation), not whether the robot succeeds — in these three demo scenarios it gets through either
way, by going around people rather than through them. To build a scenario where compliance decides
success or failure, narrow the corridor and add people until there is no gap to thread; the
`dense` recipe below is the starting point.

---

## Create your own scenario

Scenarios are plain YAML in `sim/scenarios/`. Copy one and edit it:

```bash
cp sim/scenarios/corridor.yaml sim/scenarios/my_lab.yaml
```

```yaml
name: my_lab
time_limit: 60.0

robot:
  start: [-12.0, 0.0]        # world frame, metres
  goal:  [12.0, 0.0]

# walls are [centre_x, centre_y, half_size_x, half_size_y]
walls:
  - [0.0,  2.4, 16.0, 0.15]  # a 4.8 m wide corridor, 32 m long
  - [0.0, -2.4, 16.0, 0.15]

pedestrians:
  - {pos: [4.0, 0.6], goal: [-14.0, 0.5]}       # walks toward the robot
  - {pos: [6.5, -0.8], goal: [-14.0, -0.6]}
```

Run it by name — no registration step:

```bash
python -m sim.simulate --scenario my_lab --video my_lab.mp4
```

Two extras worth knowing:

- **Waypoints.** Give a pedestrian a route instead of a straight line to their goal:
  `{pos: [8, 0], goal: [-12, 0], waypoints: [[4, 1.2], [0, -1.0], [-6, 0.5]]}`
- **Many pedestrians.** The YAML is just a list, so generate it:

  ```python
  import yaml, numpy as np
  rng = np.random.default_rng(0)
  peds = [{"pos": [float(x), float(y)], "goal": [-15.0, float(y)]}
          for x, y in zip(rng.uniform(2, 30, 40), rng.uniform(-2.0, 2.0, 40))]
  sc = yaml.safe_load(open("sim/scenarios/corridor.yaml"))
  sc.update(name="dense", pedestrians=peds, time_limit=90.0)
  yaml.safe_dump(sc, open("sim/scenarios/dense.yaml", "w"), sort_keys=False)
  ```

  ```bash
  python -m sim.simulate --scenario dense --video dense.mp4 --camera top
  ```

---

## What the simulator actually does

Each 0.1 s step:

1. Ask the controller for a velocity, given the robot pose, the goal, and every pedestrian
   position — the same call you would make on a real robot.
2. Move the robot by that velocity (kinematically — there is no gait controller to tune and
   nothing can explode).
3. Advance every pedestrian with a social-force model: pulled to their goal, pushed from each
   other, pushed from the robot (scaled by `--compliance`), pushed from walls.
4. Advance a **second copy of the crowd with no robot in it**. The gap between the two is
   `mean_ped_deviation_m` — the robot's causal effect, which you cannot measure on a real robot.
5. Write both into MuJoCo and render a frame.

## Assets

| what | where | licence |
|---|---|---|
| Unitree Go2 robot | `assets/unitree_go2/` | BSD-3-Clause, © Unitree Robotics, via [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) |
| Pedestrian | `assets/pedestrian/pedestrian.xml` | MIT (written for this repo) |
| Scene | generated into `assets/unitree_go2/_scene_generated.xml` at run time | MIT |

The pedestrian is a capsule figure rather than a scanned human on purpose: the photoreal human
models used in our research are NVIDIA Omniverse assets and are **not** redistributable, so
shipping one here would have put a licence trap in your clone.

Walls are drawn waist-high so the camera can see over them; the crowd model treats a wall as a
full barrier however tall it is drawn.

## Troubleshooting

| symptom | fix |
|---|---|
| `ImportError: No module named mujoco` | `pip install -r requirements-sim.txt` (and check your venv is active) |
| Black or empty video on a server | `export MUJOCO_GL=egl` (or `osmesa`) before running |
| `Error opening file '.../base_0.obj'` | you moved `assets/unitree_go2/` — the model needs `go2.xml` and its `assets/` folder together |
| `mode 'residual' needs weights` | `pip install huggingface_hub`, or pass `--ckpt`, or use `--mode sfm` |
| Robot reaches but the crowd scatters | expected at low `--compliance`; look at `mean_ped_deviation_m` |
| Runs slowly | drop `--video` (rendering dominates), or `--width 640 --height 360` |
| `OSError: [Errno 28] No space left on device` while installing | your `/tmp` is full (shared servers, often). `export TMPDIR=$PWD/tmp && mkdir -p $PWD/tmp` and reinstall |

## Honest limits

The pedestrians here use the same force model the controller's own physics prior assumes, so this
is the **friendly** case: your results will look better than reality. That is exactly why
`--compliance` exists, and why the paper's numbers come from a much harder benchmark. Treat this
package as a way to integrate and sanity-check the controller, not as evidence about it.
