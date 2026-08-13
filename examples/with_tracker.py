#!/usr/bin/env python3
"""
What a real tracker does to you, and how the controller handles it. Runs a scripted 20 s scene in
which people appear, vanish for a few frames, and get their IDs swapped -- the three failure modes
every pedestrian tracker has -- and prints what the controller commands through each.

    python examples/with_tracker.py

Rules the controller relies on:
  * Track IDs must be STABLE while a person is visible. History is keyed by ID; a new ID restarts
    that person's 1 s of history, and the controller falls back to constant-velocity for them.
  * Positions must be in the SAME world frame as robot_xy and the goal. Body-frame positions are
    the single most common integration bug.
  * Dropouts are fine. Missing IDs are dropped; when they come back the history refills in ~1 s.
"""
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from peroi_controller import PeRoIController  # noqa: E402


def scripted_tracker(t):
    """Returns {track_id: (x, y)} with realistic tracker noise, dropouts and an ID swap."""
    people = {}
    #  #1 walks head-on the whole time
    people[1] = (6.0 - 1.0 * t, 0.15)
    #  #2 appears at t = 4 s from the side
    if t >= 4.0:
        people[2] = (5.0 - 0.2 * (t - 4), -3.0 + 0.9 * (t - 4))
    #  #1 drops out for 3 frames around t = 8 s (occluded by #2)
    if 8.0 <= t < 8.3:
        people.pop(1, None)
    #  #3 is really #2 after an ID swap at t = 12 s
    if t >= 12.0:
        p = people.pop(2, None)
        if p is not None:
            people[3] = p
    return {k: (v[0] + np.random.normal(0, 0.02),      # tracker jitter, ~2 cm
                v[1] + np.random.normal(0, 0.02)) for k, v in people.items()}


def main():
    np.random.seed(0)
    ckpt = hf_hub_download("elmoghany/crowd-nav", "residual_predictor_k1.pt")
    ctrl = PeRoIController(ckpt, robot_radius=0.30, ped_radius=0.25, v_max=0.7)

    robot = np.array([0.0, 0.0])
    goal = (9.0, 0.0)
    dt, t = 0.1, 0.0
    events = {4.0: "person #2 appears", 8.0: "#1 occluded (3 frames)", 12.0: "#2 -> #3 ID swap"}

    print(f"{'t':>5} {'ids':>12} {'command':>16} {'nearest':>9}   note")
    while t < 20.0:
        people = scripted_tracker(t)
        vx, vy = ctrl.step(robot, goal, people, dt=dt)
        robot = robot + np.array([vx, vy]) * dt

        if abs(t - round(t)) < 1e-9 and int(round(t)) % 2 == 0:
            gaps = [float(np.linalg.norm(np.array(p) - robot)) - ctrl.rr for p in people.values()]
            note = next((v for k, v in events.items() if abs(k - t) < 0.5), "")
            print(f"{t:5.1f} {str(sorted(people)):>12} ({vx:+.2f},{vy:+.2f})   "
                  f"{min(gaps) if gaps else float('nan'):8.2f}   {note}")
        if float(np.linalg.norm(np.array(goal) - robot)) < 0.4:
            print(f"{t:5.1f}  goal reached")
            break
        t += dt

    print(f"\nfinished at ({robot[0]:.2f}, {robot[1]:.2f}); goal was {goal}")
    print("The ID swap costs ~1 s of history for that person -- the controller keeps moving because\n"
          "it falls back to constant velocity until the buffer refills. Stable IDs are worth having,\n"
          "but a swap is not a crash.")


if __name__ == "__main__":
    main()
