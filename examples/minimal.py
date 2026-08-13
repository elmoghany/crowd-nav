#!/usr/bin/env python3
"""
The whole integration, in one screen. Copy this into your stack and replace the two TODOs.

    python examples/minimal.py
"""
import os
import sys
import time

from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from peroi_controller import PeRoIController  # noqa: E402

ckpt = hf_hub_download("elmoghany/crowd-nav", "residual_predictor_k1.pt")
ctrl = PeRoIController(ckpt, robot_radius=0.30, ped_radius=0.25, v_max=0.6)

goal = (8.0, 0.0)                      # where you want the robot to end up, world frame
last = time.time()

for _ in range(200):
    now = time.time()
    dt, last = now - last, now

    robot_xy = (0.0, 0.0)              # TODO: your localization, world frame (metres)
    people = {7: (3.0, 0.4), 9: (4.1, -1.2)}   # TODO: your tracker, {track_id: (x, y)}, SAME frame

    vx, vy = ctrl.step(robot_xy, goal, people, dt=dt)   # world-frame velocity command (m/s)

    # TODO: send (vx, vy) to your base. Differential drive? Rotate into the body frame first:
    #   v = hypot(vx, vy); yaw_err = atan2(vy, vx) - robot_yaw
    #   cmd.linear.x = v * cos(yaw_err);  cmd.angular.z = k_yaw * wrap(yaw_err)
    print(f"dt={dt:5.3f}s  command=({vx:+.2f}, {vy:+.2f}) m/s")
    time.sleep(0.05)
