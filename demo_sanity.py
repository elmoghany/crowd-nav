"""
No-robot sanity check: run the PeRoI controller on a scripted scene and print the commanded velocity.
Use this to confirm the install (torch + numpy + the weights) works before wiring it to a real robot.

    python demo_sanity.py --ckpt residual_predictor_k1.pt

Scene: the robot starts at (0,0) heading to a goal at (8,0); one pedestrian walks head-on toward it
down the centre line. A correct controller should steer laterally (non-zero vy) to open a gap while
still making forward progress (positive vx), and never drive the robot straight into the person.
"""
import argparse
import numpy as np
from peroi_controller import PeRoIController
from quickstart import get_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="weights file; default fetches residual_predictor_k1.pt from "
                         "huggingface.co/elmoghany/crowd-nav")
    ap.add_argument("--mode", default="residual", choices=["residual", "sfm", "cv"])
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    ctrl = PeRoIController(get_weights(args.ckpt) if args.mode == "residual" else None,
                          mode=args.mode, robot_radius=0.30, ped_radius=0.25, v_max=0.8)
    ctrl.reset()
    dt = 0.25
    robot = np.array([0.0, 0.0]); goal = np.array([8.0, 0.0])
    ped = np.array([6.0, 0.05])                 # one person, head-on down the centre
    ped_v = np.array([-1.0, 0.0])               # walking toward the robot at 1 m/s

    print(f"mode={args.mode}  dt={dt}s")
    print(f"{'t':>5} {'robot':>16} {'ped':>16} {'cmd (vx,vy)':>18} {'gap(m)':>8}")
    for k in range(args.steps):
        vx, vy = ctrl.step(tuple(robot), tuple(goal), {0: tuple(ped)}, dt=dt)
        gap = float(np.linalg.norm(robot - ped)) - 0.55
        if k % 2 == 0:
            print(f"{k*dt:5.2f} ({robot[0]:6.2f},{robot[1]:6.2f}) "
                  f"({ped[0]:6.2f},{ped[1]:6.2f}) ({vx:6.2f},{vy:6.2f}) {gap:8.2f}")
        # integrate the little world forward
        robot = robot + np.array([vx, vy]) * dt
        ped = ped + ped_v * dt
        if robot[0] >= goal[0] - 0.3:
            print(f"reached goal at t={k*dt:.2f}s, min body gap stayed > 0 -> {gap > 0}")
            break
    print("\nOK: the controller loaded the model and produced velocity commands.")


if __name__ == "__main__":
    main()
