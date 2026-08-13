# crowd-nav — a drop-in social-navigation controller for real robots

Your robot has a goal, a position, and a list of tracked people. This gives you a velocity command
that gets to the goal **while anticipating how each person will react to the robot** — rather than
treating them as moving obstacles and freezing when they come close.

One file, no simulator, no ROS dependency, CPU-only, ~3 ms per decision.

```python
vx, vy = ctrl.step(robot_xy, goal_xy, {track_id: (x, y), ...}, dt)
```

---

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
from peroi_controller import PeRoIController

ckpt = hf_hub_download("elmoghany/crowd-nav", "residual_predictor_k1.pt")
ctrl = PeRoIController(ckpt, robot_radius=0.30, ped_radius=0.25, v_max=0.6)

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

## Background

The method: predict each person's response to the robot's *candidate* action, then choose the action
— rather than predicting people once and planning around the forecast. The research repository
(full controller with the convex-QP planner, Isaac Sim benchmarks, paper) is
[elmoghany/crowd-nav-legacy](https://github.com/elmoghany/crowd-nav-legacy); results and videos at
[elmoghany.com/crowd-nav-3](https://elmoghany.com/crowd-nav-3/).

MIT licensed. Issues and questions welcome.
