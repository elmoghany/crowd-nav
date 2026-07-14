# PeRoI controller — social navigation on a real robot

A **drop-in, simulator-free controller** that makes a mobile robot navigate a crowd *gently*: it
**anticipates how each nearby person will react to the robot** with a learned action-conditioned
predictor (a NeuRoSFM residual trained on real robot–human interaction data), and plans around them
with a short-horizon MPC. In benchmarks the same planner using this predictor perturbs the crowd
**~27% less** than using a constant-velocity or social-force model, and ~2× less than reactive
controllers (ORCA / social-force). Results + videos: **https://elmoghany.com/crowd-nav-2**.

This repo is everything you need to run it on a robot: **one dependency-light file**
(`peroi_controller.py`, needs only `torch` + `numpy`) plus the trained weights
(`residual_predictor.pt`, ~90 KB). No MuJoCo, no ROS required to use the core.

---

## 1. Install

```bash
pip install torch numpy huggingface_hub   # CPU is fine — the model is a tiny MLP, <1 ms/step
```

Then get the trained weights from the **private model repo**
[`elmoghany/peroi-controller`](https://huggingface.co/elmoghany/peroi-controller) on Hugging Face
(ask the owner for access, then `hf auth login`):

```bash
hf download elmoghany/peroi-controller residual_predictor.pt --local-dir .
```

Keep `residual_predictor.pt` next to `peroi_controller.py`.

## 2. Sanity check (no robot)

```bash
python demo_sanity.py --ckpt residual_predictor.pt
```
It runs a scripted head-on encounter and prints the commanded velocity — you should see the robot make
forward progress (`vx>0`) while steering aside (`vy≠0`) to open a gap, never driving into the person.
This confirms `torch`, `numpy`, and the weights load correctly.

## 3. Wire it into your control loop

```python
from peroi_controller import PeRoIController

ctrl = PeRoIController(
    "residual_predictor.pt",
    robot_radius=0.30,   # your robot's footprint radius (m)
    ped_radius=0.25,     # assumed person radius (m)
    v_max=0.6,           # START LOW for first real tests (m/s)
)
ctrl.reset()

while running:
    # ALL in one fixed world frame (your SLAM/`map` frame), metres:
    robot_xy = (rx, ry)                                   # robot position from localization
    goal_xy  = (gx, gy)                                   # where you want it to go
    peds     = {track_id: (px, py) for ...}              # tracked people (dict of id -> (x,y))

    vx, vy = ctrl.step(robot_xy, goal_xy, peds, dt=loop_dt)   # desired velocity, WORLD frame (m/s)
    send_to_base(vx, vy)                                  # see "Command types" below
```

### The contract
- **One world frame.** Robot pose, goal, and every pedestrian position must be in the **same fixed
  frame** (e.g. `map`), in **metres**. Don't mix base-frame and map-frame.
- **Pedestrian tracks.** Pass `{track_id: (x, y)}`. IDs must be **stable across calls** (that's how the
  predictor accumulates each person's 1 s of history). Any tracker that outputs IDs works
  (SPENCER, ZED/OAK people, a LiDAR leg/DR-SPAAM tracker, ...). A plain `list[(x,y)]` also works **iff**
  your tracker keeps a stable order.
- **Rate.** Call `step()` at **≥ 4 Hz** and pass the real `dt` (seconds since your last call). Internally
  it samples/replans every **0.25 s** and holds the command between ticks, so calling faster is fine and
  just makes the held command fresher. The predictor observes 1 s of history and predicts 2 s ahead.
- **Output.** `(vx, vy)` is a **world-frame** velocity. Convert per your base:

### Command types
- **Holonomic base** (can strafe): rotate the world-frame `(vx,vy)` into the base frame by the robot yaw:
  `vx_base = cos(-yaw)*vx - sin(-yaw)*vy`, `vy_base = sin(-yaw)*vx + cos(-yaw)*vy` → `geometry_msgs/Twist`.
- **Differential-drive base** (can't strafe): convert to `(v, omega)` — drive forward, turn toward the
  desired direction. `ros_node.py::to_diff_drive()` does exactly this.
- A ready-to-adapt **ROS 1 / ROS 2 node** is in `ros_node.py` (subscribes odom + tracked people + goal,
  publishes `/cmd_vel`). The two things you customize are the tracked-people callback and the base type.

## 4. Safety (read this)

This is a **social-navigation planner, not a safety layer.** Keep underneath it:
- your platform **e-stop** and **collision monitor** (bumpers / safety LiDAR),
- a hard **speed cap** (`v_max`) — start at 0.4–0.6 m/s,
- your own **static-obstacle avoidance** (walls, furniture): PeRoI plans around *people*, not the map.
  Either run it on top of your local planner/costmap, or pass a corridor (below) for simple hallways.

## 5. Tuning

| arg | meaning | default |
|---|---|---|
| `v_max` | max commanded speed (m/s) | 0.8 |
| `comfort` | personal-space radius it tries to keep (m) | 0.8 |
| `safety` | hard min surface clearance; candidates closer are rejected (m) | 0.30 |
| `consider_r` / `k_near` | radius / max number of people it plans around | 4.5 m / 6 |
| `w_clear` / `w_progress` / `w_smooth` | cost weights: clearance vs goal progress vs smoothness | 6 / 1 / 0.3 |
| `mode` | `"residual"` (trained, default), `"sfm"`, or `"cv"` — same planner, different anticipation | residual |

Optional extras:
- **Hazard zones** — pass `hazards=[[[x,y],...], ...]` (polygons in world frame) to keep the robot out
  of, and to avoid herding people into, floor hazards (potholes, door thresholds).
- **Corridor keep-in** — pass `half_width=` (m) and `corridor_axis=(x,y)` (unit vector along the hall)
  for a simple straight corridor, so the planner keeps the robot inside it.

## 6. What's in here
```
peroi_controller.py    # the controller (self-contained: predictor + SFM + MPC). torch + numpy only.
demo_sanity.py         # no-robot check
ros_node.py            # ROS 1/2 example node
requirements.txt
```
The trained weights (`residual_predictor.pt`, ~90 KB) are hosted separately on the private Hugging Face
repo **[elmoghany/peroi-controller](https://huggingface.co/elmoghany/peroi-controller)** — download them
as shown in step 1.

## Model card / provenance
The predictor is a NeuRoSFM residual (`ŷ = SocialForce + learned_correction`) trained on the real
**PeRoI** robot–human interaction dataset (3 robot conditions, per-person avoidance/neutral/attraction
labels). On held-out real recordings it cuts pedestrian-prediction ADE ~20% over constant-velocity and
~43% over social-force, and its predicted robot-effect is larger for people humans labelled *influenced*
(a causal check). Full write-up, metrics, and per-metric videos: **https://elmoghany.com/crowd-nav-2**.
Known limitation: trained on slow-robot data, so it is out-of-distribution (degrades) in very dense
fast crowds — keep `v_max` modest and your safety layer active.

Questions: Mohamed Elmoghany (Cornell). License: MIT.
