#!/usr/bin/env python3
"""
Zero-setup check that the controller works on THIS machine. Downloads the weights, runs two
scripted scenes with no simulator, and tells you whether the output looks right.

    pip install -r requirements.txt
    python quickstart.py

Nothing here touches a robot. The "pedestrians" walk blindly at constant velocity -- a real crowd
yields, so treat the clearances below as a pessimistic floor.
"""
import argparse
import sys

import numpy as np

from peroi_controller import PeRoIController

HF_REPO = "elmoghany/crowd-nav"
WEIGHTS = "residual_predictor_k1.pt"


def get_weights(path=None):
    """Local path if given, else pull the weights from the Hugging Face repo (cached after once)."""
    if path:
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is missing. Either `pip install huggingface_hub`, or download\n"
                 f"  https://huggingface.co/{HF_REPO}/resolve/main/{WEIGHTS}\n"
                 "and pass it with --ckpt.")
    print(f"[quickstart] fetching {WEIGHTS} from {HF_REPO} ...")
    return hf_hub_download(HF_REPO, WEIGHTS)


def run_scene(ctrl, name, robot, goal, peds, ped_vels, seconds=14.0, dt=0.1):
    """Integrate the robot with the controller's commands; pedestrians walk on rails."""
    ctrl.reset()
    r = np.array(robot, float)
    P = {i: np.array(p, float) for i, p in enumerate(peds)}
    V = {i: np.array(v, float) for i, v in enumerate(ped_vels)}
    min_clear, path_len, t = 1e9, 0.0, 0.0

    print(f"\n  {name}")
    print(f"  {'t':>5} {'robot':>16} {'command':>16} {'nearest person':>16}")
    while t < seconds:
        vx, vy = ctrl.step(r, goal, P, dt=dt)
        r = r + np.array([vx, vy]) * dt
        path_len += float(np.hypot(vx, vy)) * dt
        for i in P:
            P[i] = P[i] + V[i] * dt
        gaps = [float(np.linalg.norm(p - r)) - ctrl.rr for p in P.values()]
        min_clear = min(min_clear, min(gaps))
        if abs((t / 2.0) - round(t / 2.0)) < 1e-9:                    # every 2 s
            j = int(np.argmin(gaps))
            print(f"  {t:5.1f} ({r[0]:5.2f},{r[1]:5.2f}) ({vx:5.2f},{vy:5.2f}) "
                  f"{gaps[j]:9.2f} m away")
        if float(np.linalg.norm(np.array(goal) - r)) < 0.4:
            print(f"  {t:5.1f}  goal reached")
            break
        t += dt

    reached = float(np.linalg.norm(np.array(goal) - r)) < 0.4
    print(f"  -> reached={reached}  min surface clearance={min_clear:+.2f} m  path={path_len:.1f} m")
    return reached, min_clear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="local weights file (default: fetch from HF)")
    ap.add_argument("--v_max", type=float, default=0.8, help="commanded speed cap (m/s)")
    args = ap.parse_args()

    ctrl = PeRoIController(get_weights(args.ckpt), robot_radius=0.30, ped_radius=0.25,
                           v_max=args.v_max, mode="residual")
    print(f"[quickstart] controller ready - v_max={args.v_max} m/s, "
          f"clearance target {ctrl.safety:.2f} m beyond bodies")

    ok_head, clr_head = run_scene(
        ctrl, "HEAD-ON - one person walking straight at the robot",
        robot=(0.0, 0.0), goal=(8.0, 0.0),
        peds=[(6.0, 0.0)], ped_vels=[(-1.1, 0.0)])

    ok_cross, clr_cross = run_scene(
        ctrl, "CROSSING - three people cutting across the robot's path",
        robot=(0.0, 0.0), goal=(9.0, 0.0),
        peds=[(4.0, -3.0), (5.2, -3.6), (6.4, -3.2)],
        ped_vels=[(0.15, 0.95), (0.10, 1.0), (0.05, 0.9)])

    print("\n  verdict")
    good = clr_head > 0.0 and clr_cross > 0.0 and ok_head
    for label, ok, clr in (("head-on", ok_head, clr_head), ("crossing", ok_cross, clr_cross)):
        print(f"    {label:9s} clearance {clr:+.2f} m  reached={ok}")
    if good:
        print("    install OK - the controller anticipates and keeps a gap.")
    else:
        print("    something is off: clearance should stay positive. Check that the weights match\n"
              "    the deployment prior (see README, 'Which weights').")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
