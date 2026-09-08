# crowd-nav — a drop-in social-navigation controller for real robots

Your robot has a goal, a position, and a list of tracked people. This gives you a velocity command
that gets to the goal **while anticipating how each person will react to the robot** — rather than
treating them as moving obstacles and freezing when they come close.

One file, no simulator, no ROS dependency, CPU-only, ~3 ms per decision.

```python
vx, vy = ctrl.step(robot_xy, goal_xy, {track_id: (x, y), ...}, dt)
```

---

## Two ways in

| you want to | go to |
|---|---|
| **run a simulation** — robot + crowd + video, no hardware | **[SIMULATION.md](SIMULATION.md)** (command by command, ~5 min) |
| run it from Windows, or on a Linux server and watch live | **[REMOTE.md](REMOTE.md)** |
| put the controller on a real robot | this page |
| **reproduce the paper's videos** — 90 pedestrians, Isaac Sim, a GPU cluster | [**Reproduce the paper's videos**](#reproduce-the-papers-videos), and [what is missing](#what-is-missing-and-what-needs-replacing) |

## 60-second start

```bash
git clone https://github.com/elmoghany/crowd-nav && cd crowd-nav
pip install -r requirements.txt huggingface_hub
python quickstart.py
```

`quickstart.py` downloads the weights, runs two scripted scenes with no robot and no simulator, and
tells you whether the output looks right. Expected result:

```
  HEAD-ON - one person walking straight at the robot
    0.0 ( 0.08, 0.00) ( 0.80, 0.00)      5.26 m away
    2.0 ( 1.29,-0.70) ( 0.49,-0.56)      1.95 m away     <- steers aside early
    4.0 ( 2.62,-1.27) ( 0.78, 0.18)      1.15 m away
   10.5  goal reached
  -> reached=True  min surface clearance=+0.76 m  path=8.4 m

  verdict
    head-on   clearance +0.76 m  reached=True
    crossing  clearance +1.15 m  reached=True
    install OK - the controller anticipates and keeps a gap.
```

The lateral command appears **while the person is still 2 m away** — that is the anticipation doing
its job. A reactive planner produces zero lateral motion until the gap closes, then swerves or stops.

---

## Use it in your stack

The entire integration, from `examples/minimal.py`:

```python
from huggingface_hub import hf_hub_download
from crowd_nav import CrowdNavController as Controller   # or: from peroi_controller import PeRoIController

ckpt = hf_hub_download("elmoghany/crowd-nav", "residual_predictor_k1.pt")
ctrl = Controller(ckpt, robot_radius=0.30, ped_radius=0.25, v_max=0.6)

goal = (8.0, 0.0)                       # world frame, metres

while running:
    robot_xy = my_localization()        # (x, y) world frame
    people   = my_tracker()             # {track_id: (x, y)}, SAME frame
    vx, vy   = ctrl.step(robot_xy, goal, people, dt=elapsed_since_last_call)
    send_to_base(vx, vy)                # world-frame velocity, m/s
```

**Differential-drive bases** need the command rotated into the body frame:

```python
import math
v = math.hypot(vx, vy)
yaw_err = math.atan2(vy, vx) - robot_yaw
yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi     # wrap to [-pi, pi]
cmd.linear.x  = v * math.cos(yaw_err)                        # back off while turning
cmd.angular.z = 1.5 * yaw_err
```

**ROS 1 or ROS 2**: `ros_node.py` is a working reference node — subscribe your odometry, your
tracker, and a goal topic; it publishes `geometry_msgs/Twist`. Adapt two callbacks and you are done.

**Real trackers drop and swap IDs.** `examples/with_tracker.py` runs a scene with jitter, a 3-frame
dropout and an ID swap so you can see exactly what the controller does through each (short answer:
it keeps moving; a new ID just costs that person ~1 s of history).

---

## Which weights

| file | use it when |
|---|---|
| **`residual_predictor_k1.pt`** | **default.** Physics prior + learned correction, matched to this controller's deployment prior. |
| `residual_predictor.pt` | never — kept for reproducibility only. Trained against a *zero* robot-force prior while this controller deploys at 1.0, so the correction fixes an error that is not there. It was a real bug of ours; `_k1` is the fix. |
| `linres_head.pt` | not used by this package. It is the strongest model in our research stack, but its advantage (an exactly-linear response, enabling a convex QP planner) only pays off with that planner — this package ships the simpler velocity-grid planner. |

All three: [huggingface.co/elmoghany/crowd-nav](https://huggingface.co/elmoghany/crowd-nav).

---

## Tuning for your robot

Start conservative and raise `v_max` once you trust it.

| argument | default | what it does |
|---|---|---|
| `v_max` | `0.8` | speed cap (m/s). **Start at 0.4–0.6 on a real robot.** |
| `robot_radius`, `ped_radius` | `0.30`, `0.25` | body radii; clearance is measured surface-to-surface |
| `safety` | `0.30` | extra clearance the planner tries to keep beyond the two bodies (m) |
| `comfort` | `0.8` | distance (m) under which closeness starts costing — raise it to be more polite |
| `mode` | `"residual"` | `"sfm"` (physics only) or `"cv"` (straight lines) — same planner, dumber anticipation. Useful for A/B-ing whether the learned model is helping on *your* robot |
| `hazards` | `None` | list of polygons `[[x,y],...]` to avoid and not herd people into (stairs, docks, roads) |
| `half_width`, `corridor_axis` | `None` | keep-in for a known corridor |

```python
ctrl = PeRoIController(ckpt, v_max=0.5, safety=0.40, comfort=1.0,
                       hazards=[[[2,1],[4,1],[4,3],[2,3]]])      # a stairwell, world frame
```

---

## What it actually does each cycle

1. Keeps 1 second of history for the 6 nearest people within 4.5 m.
2. Rolls out ~20 candidate velocities for the robot, 2 seconds ahead.
3. For each candidate, predicts **how those people would respond to that specific robot motion** —
   a social-force prior plus a neural correction trained on real robot–pedestrian recordings.
4. Scores candidates on progress, clearance, comfort, smoothness (plus walls/hazards) and commands
   the winner. Repeats at 4 Hz.

Step 3 is the part that differs from a normal local planner: the people in the forecast **react to
the plan being considered**, so the controller can prefer a path that opens a gap instead of one
that closes it.

---

## Troubleshooting

| symptom | cause |
|---|---|
| Robot freezes in a crowd | `v_max` too low for the density, or `safety`/`comfort` too large. Also check `dt` — passing a wrong `dt` breaks the internal 0.25 s clock. |
| Commands look mirrored | People are in the **body frame**, not the world frame. Everything you pass must share one frame. |
| Jerky commands | Your tracker IDs are unstable, so histories keep resetting. Check with `examples/with_tracker.py` as a reference for what stable-ID behaviour looks like. |
| No lateral anticipation | You are on `mode="cv"`, or the weights failed to load — the constructor raises if `mode="residual"` and no checkpoint is given, so check startup logs. |
| Works in `quickstart.py`, not on the robot | Almost always frames or `dt`. Log `robot_xy`, one person's `(x, y)` and `dt` for 2 seconds and sanity-check by hand. |

**This is a local planner, not a safety system.** It has no formal collision guarantee, no emergency
stop, and no obstacle avoidance for anything that is not a tracked person. Keep your own safety
layer (bumper, lidar e-stop, speed limiter) underneath it.

---

## Reproduce the paper's videos

The five clips on [elmoghany.com/crowd-nav-3](https://elmoghany.com/crowd-nav-3/) — the Go2 crossing
a 4.5 m × 105 m corridor against **90 oncoming pedestrians**, with the robot's plan polyline, its
predicted-vs-counterfactual future for each person, and a live HUD drawn on every frame. About an
hour, most of it the Isaac install.

> **Read this first.** These steps do **not** run from this package. They run from the research
> repository, which is a separate, larger codebase (Isaac Sim environment, overlay renderer, convex-QP
> planner, scenario generator). This package ships the controller and a lightweight MuJoCo simulator
> only — see [SIMULATION.md](SIMULATION.md) for what *does* run here in five minutes.
> [**What is missing, and what needs replacing**](#what-is-missing-and-what-needs-replacing) lists
> every piece you have to supply.

**You need:** a SLURM cluster with an NVIDIA GPU whose driver is **≥ 580**; ~20 GB of disk; Python
3.12; and access to the research repo. Nothing for the NVIDIA assets — the stock Go2 and the human
meshes stream from NVIDIA's public S3 at run time.

### 1. Get the code

```bash
git clone https://github.com/elmoghany/crowd-nav-legacy && cd crowd-nav-legacy
export PATH=/usr/local/slurm/current/bin:$PATH      # never run compute on the login node
```

### 2. Build the Isaac environment

```bash
sbatch isaac_env/cluster_setup/build_isaac6.sbatch   # edit the paths at the top first
```

Python 3.12 is not optional — Isaac 6.x wheels are py312-only. If it fails with a bare
`No solution found`, that is `isaacsim-core` pinning a **pre-release** dependency
(`tinyobjloader==2.0.0rc13`) which `uv` excludes by default:

```bash
sbatch isaac_env/cluster_setup/fix_isaac6.sbatch     # same install with --prerelease=allow
```

Result: `.venv-isaac`. Confirm with `.venv-isaac/bin/python -c "import isaacsim; print('ok')"`.

### 3. Bake the standing robot (once)

Both stock Go2 assets are authored **splayed** — feet 0.449 m below the origin — so rendered as-is
the robot looks like a frog lying flat. The environment refuses to render them rather than produce a
wrong picture. The bake lets PhysX settle the robot onto its feet and freezes that pose into an
override layer:

```bash
sbatch --gres=gpu:1 --wrap '.venv-isaac/bin/python isaac_env/bake_go2_stand.py'
ls -la assets/go2_usd/go2_standing.usda      # ~125 MB when it worked
```

### 4. Fetch the weights

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs("results/_causal_k1", exist_ok=True)
os.makedirs("results/_foresight_models", exist_ok=True)
shutil.copy(hf_hub_download("elmoghany/crowd-nav", "residual_predictor_k1.pt"),
            "results/_causal_k1/residual_predictor_k1.pt")
shutil.copy(hf_hub_download("elmoghany/crowd-nav", "linres_head.pt"),
            "results/_foresight_models/linres_head.pt")
print("weights in place")
PY
```

### 5. Generate the scenario

The 90-pedestrian corridor is generated from a seed, not hand-written:

```bash
python runner/gen_np1_50.py
# wrote scenarios/np1_50_narrow_passage.yaml: 90 peds, start-x span [-53.9, 248.5],
# 90 oncoming / 0 co-flow
```

50 pedestrians spread through the hall plus 40 late entrants staged outside the far mouth so the
corridor never empties. Every route segment is verified against the walls at a 0.30 m margin.

### 6. Pick a node that can actually render

**This is the step people lose a day to.** The RTX renderer is gated on the *driver*, which varies
per node, and a too-old driver returns **zero frames silently** — no exception, just an empty video.
There is no node-picking script in either repo (see the notes below); check the node yourself before
booking a long job:

```bash
srun -p <partition> -w <node> --gres=gpu:1 --time=00:01:00 \
     nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

Driver ≥ 580 is the bar. Two failure modes are not driver-related and cost a full wall-clock limit
each: a node with a good driver but a broken render stack (`Sensor beginFrame() failed`, then the
job hangs), and a node reporting a wrong GPU count. `isaac_env/CLUSTER_NODES.md` in the research
repo records which of *our* nodes fall into each bucket — that list is site-specific and will not
transfer to your cluster.

### 7. Render the five clips

One job per clip, run them in parallel:

```bash
for CFG in "foresight ours" "foresight_linres ours" "foresight srfm" "foresight sfm" "foresight orca"; do
  set -- $CFG
  sbatch --nodelist=<YOUR RENDER NODE> --export=ALL,METHOD=$1,CROWD=$2 isaac_env/fs50_render.sbatch
done
```

`METHOD` is the robot's brain (`foresight`, `foresight_linres`, `foresight_conf`, `foresight_grid`,
`foresight_mppi`); `CROWD` is the pedestrian model (`ours`, `srfm`, `sfm`, `orca`) — a different
axis. ~25 min per clip. Output: `results/_isaac/ovl50_<method>_<crowd>.mp4` + a `.json` of metrics.

Prefer one interactive allocation if you are iterating — it avoids paying Isaac's 2–4 minute boot
and the queue wait on every run:

```bash
salloc -p <partition> --nodelist=<YOUR RENDER NODE> --gres=gpu:1 \
       --cpus-per-task=6 --mem=28G --time=02:00:00
```

### 8. Check the clips before believing them

Non-negotiable here, and it has caught four separate silent failures:

```bash
python - <<'PY'
import imageio.v2 as im, glob
for f in sorted(glob.glob("results/_isaac/ovl50_*.mp4")):
    fr = [x for x in im.get_reader(f)]
    miss = []
    for i in range(0, len(fr), 20):                      # ~1 frame per second
        a = fr[i].astype(int)
        orange = ((a[:,:,0] > 150) & (a[:,:,1] > 80) & (a[:,:,1] < 190) & (a[:,:,2] < 90)).sum()
        if orange < 50: miss.append(i // 20)
    print(f, len(fr), "frames | seconds with no robot:", miss or "none")
PY
```

The robot is tinted orange precisely so this check is possible. If it vanishes for a stretch you are
looking at a rendering failure, not a controller that gave up.

### 9. What you should get

| clip | outcome |
|---|---|
| `ovl50_foresight_linres_ours` | reached, 98.5 s, **zero contacts** |
| `ovl50_foresight_srfm` | reached, 85.0 s, zero contacts |
| `ovl50_foresight_sfm` | reached, 91.7 s, zero contacts |
| `ovl50_foresight_ours` | reached, 90.0 s, 1 hard contact |
| `ovl50_foresight_orca` | reached, 96.7 s, 2 contacts |

Timings shift a little with Isaac version and node, but reach and the ordering (ORCA hardest, LinRes
cleanest) should hold. If everything suddenly looks perfect, check the crowd is actually oncoming — a
scenario regenerated with different parameters is the usual cause.

### 10. When it goes wrong

| symptom | cause |
|---|---|
| Empty or black video, no error | driver too old on that node — step 6 |
| `Sensor beginFrame() failed`, then the job hangs | that node's render stack is broken; use another |
| Robot invisible or washed white | the floor slab does not span the scene, so the robot renders over void and auto-exposure blows out. If you extend the hall, extend the slab too |
| Clip shows a frozen backdrop under a live overlay | renderer stalled; the runner raises after ~6 s of identical frames rather than writing a corrupt clip |
| `no baked standing Go2 at ...` | step 3 |
| Job runs to its wall limit with no output | the long corridor hangs the render loop with slow policies (e.g. SICNav at ~800 ms/plan). Render those on a doorway scenario instead |
| Env var list truncated in a job | SLURM `--export=ALL,VAR=a,b` splits on the comma. Hardcode lists inside the sbatch |
| `No space left on device` | `/tmp` is not shared with compute nodes and the login node's fills up. Put scratch on shared storage and set `TMPDIR` |
| Run ids show `nogit` | export `CROWDNAV_GIT_SHA` — a copied tree is not a git checkout |
| Cannot open a viewer from your laptop | high ports are usually firewalled (22 is open); tunnel through the login node — see [REMOTE.md](REMOTE.md). Isaac's WebRTC livestream cannot work from a compute node at all: its media is UDP, and SSH forwards TCP |

---

## What is missing, and what needs replacing

Everything below is **not in this repository**. Some of it is deliberate (size, third-party
licences), some of it is work still to be done. Read this as the to-do list for making the video
flow above runnable from this repo alone.

### Removed on purpose — you must supply your own

| what | why it went | what to do instead |
|---|---|---|
| **Free-GPU node picker** (`~/unicorn-gpus.sh`) | It lived in one person's home directory, was never in any repo, and hard-coded our cluster's node names and partitions. Every doc reference to it has been removed. | Query your own scheduler (`sinfo -O nodehost,gres,gresused,statelong -p <partition>`), then confirm the driver with the one-liner in step 6. |
| **Site-specific node allow/deny list** | The good/bad node table was ours, not yours — copying it would send you to machines that do not exist. | Build your own the first time: run the step-6 driver check across your GPU partition and record which nodes render. |

### Not in git by design

| what | size | how you get it |
|---|---|---|
| Isaac Sim 6.0.1 | ~15 GB | installed by step 2 |
| Stock Unitree Go2 USD, skinned human USDs | — | streamed from NVIDIA's public S3 at run time; NVIDIA's assets, not ours to redistribute |
| Baked standing Go2 (`assets/go2_usd/go2_standing.usda`) | ~125 MB | you generate it in step 3; derived from NVIDIA's asset |
| Model weights | ~210 KB | [huggingface.co/elmoghany/crowd-nav](https://huggingface.co/elmoghany/crowd-nav), fetched in step 4 |

### Lives in the research repo, not here — the actual porting to-do

The video flow depends on code this package does not ship. To make it reproducible from a clone of
*this* repo, these are the pieces that would have to move over:

| piece | what it is | note |
|---|---|---|
| `isaac_env/` | Isaac Sim environment, overlay renderer, the `fs50_render.sbatch` job, the bake and setup scripts | the bulk of the work; ~40 files |
| `runner/gen_np1_50.py` | seeded generator for the 90-pedestrian corridor scenario | small and self-contained |
| Convex-QP planner (`control/scp_qp.py`, `policies/foresight_mpc.py`) | the `foresight` / `foresight_linres` methods used in the clips | this package ships the simpler velocity-grid planner, so `linres_head.pt` has nothing to drive here |
| Pedestrian models `srfm`, `orca`, `ours` | the `CROWD` axis of the render matrix | this package's simulator ships one social-force crowd model |
| `isaac_env/CLUSTER_NODES.md` | measured driver/render notes per node | site-specific; would need rewriting generically |

The research repository is [`elmoghany/crowd-nav-legacy`](https://github.com/elmoghany/crowd-nav-legacy)
and is **private** — request access if you need it. Weights and results are public.

### Known gaps in this package

| gap | status |
|---|---|
| No unit tests beyond the CI import + physics-mode smoke test | the controller's behaviour is checked by `quickstart.py`'s two scripted scenes only |
| No ROS 2 package | `ros_node.py` is a reference node, not an installable package (no `package.xml`, no launch file) |
| Simulator is MuJoCo-only | the Isaac path is research-repo-only, as above |
| `linres_head.pt` ships but is unusable here | needs the convex-QP planner listed above |
| Crowd-behaviour models — **published, but not used here** | the seven checkpoints under [`crowd_models/`](https://huggingface.co/elmoghany/crowd-nav/tree/main/crowd_models) drive the *pedestrians* in a separate study (our Social-LSTM / Social-GAN / transformer re-implementations, trained on pooled ETH/UCY and on JRDB). The controller and the videos above never load them; the code that does is `humanmodel/` in the research repo. |

---

## Background

The method: predict each person's response to the robot's *candidate* action, then choose the action
— rather than predicting people once and planning around the forecast. The research repository
(full controller with the convex-QP planner, Isaac Sim benchmarks, paper) is
[elmoghany/crowd-nav-legacy](https://github.com/elmoghany/crowd-nav-legacy); results and videos at
[elmoghany.com/crowd-nav-3](https://elmoghany.com/crowd-nav-3/).

**Citing this work:** see `CITATION.cff`. **Contributing:** CI runs an import + physics-mode
smoke test on Python 3.9 and 3.12 for every push, so a PR that breaks the API fails immediately.

MIT licensed. Issues and questions welcome.
