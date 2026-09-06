"""Run one closed-loop episode: the controller drives the Go2 through a simulated crowd.

    python -m sim.simulate --scenario corridor --video out.mp4

Kinematic by design (see sim/__init__.py): every frame we ask the controller for a velocity,
integrate the robot pose, advance the crowd, write both into MuJoCo and call mj_forward. No
gait tuning, no physics explosions, deterministic for a given seed.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import scene as scene_mod          # noqa: E402
from sim.crowd import Crowd                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN_DIR = os.path.join(HERE, "scenarios")


def load_scenario(name):
    path = name if os.path.exists(name) else os.path.join(SCEN_DIR, name + ".yaml")
    if not os.path.exists(path):
        avail = ", ".join(sorted(f[:-5] for f in os.listdir(SCEN_DIR) if f.endswith(".yaml")))
        raise SystemExit("scenario '" + name + "' not found. Available: " + avail +
                         "\n(or pass a path to your own .yaml -- see SIMULATION.md)")
    with open(path) as fh:
        return yaml.safe_load(fh)


def make_controller(args):
    """The controller under test. mode='residual' needs weights; 'sfm'/'cv' need none."""
    from crowd_nav import CrowdNavController
    ckpt = None
    if args.mode == "residual":
        if args.ckpt:
            ckpt = args.ckpt
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise SystemExit(
                    "mode 'residual' needs weights. Either `pip install huggingface_hub`, or pass\n"
                    "--ckpt path/to/residual_predictor_k1.pt, or run --mode sfm (no weights).")
            ckpt = hf_hub_download("elmoghany/crowd-nav", "residual_predictor_k1.pt")
    return CrowdNavController(ckpt, mode=args.mode, robot_radius=args.robot_radius,
                              ped_radius=0.25, v_max=args.v_max)


def leg_pose(t, speed):
    """A simple trot so the robot does not look like it is skating. Purely cosmetic."""
    amp = min(0.35, 0.10 + 0.30 * speed)
    ph = 2 * math.pi * (t / 0.45)
    out = {}
    for k, leg in enumerate(("FL", "RR", "FR", "RL")):
        s = math.sin(ph + (math.pi if k >= 2 else 0.0))
        out[leg + "_hip_joint"] = 0.0
        out[leg + "_thigh_joint"] = 0.9 + amp * s
        out[leg + "_calf_joint"] = -1.8 - amp * 0.9 * s
    return out


def run(args):
    import mujoco

    sc = load_scenario(args.scenario)
    peds = sc["pedestrians"]
    xml = scene_mod.build(sc)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    free_adr = int(model.jnt_qposadr[0])                 # the Go2's free joint is joint 0

    # Tint the robot high-visibility orange. The vendored Go2 is dark grey and vanishes against
    # the floor at a distance; clearing matid lets geom_rgba take effect.
    root_body = int(model.jnt_bodyid[0])
    for g in range(model.ngeom):
        if int(model.body_rootid[model.geom_bodyid[g]]) == int(model.body_rootid[root_body]):
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [1.0, 0.62, 0.09, 1.0]
    ped_bid = [model.body("ped" + str(i)).id for i in range(len(peds))]
    jnt_adr = {}
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm and nm.endswith("_joint"):
            jnt_adr[nm] = int(model.jnt_qposadr[j])

    ctrl = make_controller(args)
    start = np.array(sc["robot"]["start"], float)
    goal = np.array(sc["robot"]["goal"], float)
    walls = sc.get("walls", [])
    has_routes = any("waypoints" in p for p in peds)
    routes = [p.get("waypoints") for p in peds] if has_routes else None

    def new_crowd():
        return Crowd([p["pos"] for p in peds], [p.get("goal", p["pos"]) for p in peds],
                     walls=walls, compliance=args.compliance, seed=args.seed, waypoints=routes)

    crowd = new_crowd()
    ghost = new_crowd()          # same crowd, robot deleted: the difference is the robot's effect

    rr = args.robot_radius + 0.25
    robot = start.copy()
    yaw, dt, t = 0.0, args.dt, 0.0
    frames, plan_ms, dev_trace = [], [], []
    min_clear, contacts, reached = 1e9, 0, False
    renderer, cam, server = None, None, None
    if args.serve:
        from sim.stream import FrameServer
        server = FrameServer(args.serve)
        print("live view:")
        for u in server.urls():
            print("   ", u)
        print("    (from another machine use the host/IP line; see REMOTE.md)")
        print()
    if args.video or args.serve:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # A tracking 3/4 view. "top" looks straight down, which is the clearest way to read
        # who yields to whom; "follow" rides behind the robot and shows the gait.
        preset = {"follow": (90.0, -20.0, 7.5), "side": (90.0, -32.0, 13.0),
                  "top": (90.0, -80.0, 15.0)}[args.camera]
        cam.azimuth, cam.elevation, cam.distance = preset

    limit = float(sc.get("time_limit", args.time_limit))
    frame_every = max(1, int(round((1.0 / args.fps) / dt)))
    step_i = 0

    while t < limit:
        t0 = time.perf_counter()
        vx, vy = ctrl.step(robot, goal, {i: tuple(p) for i, p in enumerate(crowd.p)}, dt=dt)
        plan_ms.append((time.perf_counter() - t0) * 1e3)

        robot = robot + np.array([vx, vy]) * dt
        sp = float(np.hypot(vx, vy))
        if sp > 1e-3:
            yaw = math.atan2(vy, vx)
        crowd.step(robot, dt)
        ghost.step(None, dt)
        # deviation accumulated over the WHOLE episode: the final-instant gap washes out once
        # people have walked past the robot and re-converged on their goals
        dev_trace.append(float(np.linalg.norm(crowd.p - ghost.p, axis=1).mean()))

        gaps = np.linalg.norm(crowd.p - robot, axis=1) - rr
        min_clear = min(min_clear, float(gaps.min()))
        contacts += int((gaps < 0).sum())

        data.qpos[free_adr:free_adr + 3] = [robot[0], robot[1], args.stand_height]
        data.qpos[free_adr + 3:free_adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        for nm, val in leg_pose(t, sp).items():
            if nm in jnt_adr:
                data.qpos[jnt_adr[nm]] = val
        for i, bid in enumerate(ped_bid):
            model.body_pos[bid] = [crowd.p[i][0], crowd.p[i][1], 0.0]
        mujoco.mj_forward(model, data)

        if renderer is not None and step_i % frame_every == 0:
            cam.lookat[:] = [robot[0], robot[1], 0.8]
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            if args.video:
                frames.append(rgb)
            if server is not None:
                server.publish(rgb)
                if args.realtime:
                    time.sleep(max(0.0, (1.0 / args.fps) - (time.perf_counter() - t0)))

        if float(np.linalg.norm(goal - robot)) < 0.5:
            reached = True
            break
        t += dt
        step_i += 1

    deviation = float(np.mean(dev_trace)) if dev_trace else 0.0
    peak_dev = float(np.max(dev_trace)) if dev_trace else 0.0
    res = {
        "scenario": sc.get("name", args.scenario), "mode": args.mode,
        "reached": bool(reached), "sim_time_s": round(t, 2),
        "min_clearance_m": round(min_clear, 3), "contact_steps": int(contacts),
        "mean_ped_deviation_m": round(deviation, 3),
        "peak_ped_deviation_m": round(peak_dev, 3),
        "plan_ms_mean": round(float(np.mean(plan_ms)), 2),
        "plan_ms_p95": round(float(np.percentile(plan_ms, 95)), 2),
        "pedestrians": len(peds), "compliance": args.compliance,
    }
    if server is not None:
        print()
        print("  simulation finished -- the live view freezes on the last frame.")
        if args.hold:
            print("  holding the page open for {}s (Ctrl-C to stop)".format(args.hold))
            time.sleep(args.hold)
        server.close()
    if renderer is not None:
        if frames:
            import imageio.v2 as imageio
            imageio.mimsave(args.video, frames, fps=args.fps, macro_block_size=None,
                            codec="libx264", quality=7)
            res["video"] = args.video
            res["frames"] = len(frames)
        renderer.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="corridor", help="name in sim/scenarios/ or a .yaml path")
    ap.add_argument("--mode", default="sfm", choices=["residual", "sfm", "cv"],
                    help="controller's anticipation model ('sfm'/'cv' need no weights)")
    ap.add_argument("--ckpt", default=None, help="weights for --mode residual")
    ap.add_argument("--video", default="", help="write an mp4 here (omit for a metrics-only run)")
    ap.add_argument("--camera", default="top", choices=["top", "side", "follow"],
                    help="top = bird's eye (clearest); side = 3/4 view; follow = behind the robot")
    ap.add_argument("--compliance", type=float, default=1.0,
                    help="how much the crowd yields: 1.0 cooperative, 0.1 stubborn")
    ap.add_argument("--v_max", type=float, default=0.8)
    ap.add_argument("--robot_radius", type=float, default=0.30)
    ap.add_argument("--stand_height", type=float, default=0.33)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serve", type=int, default=0, metavar="PORT",
                    help="stream the run to a browser on this port (watch a remote run live)")
    ap.add_argument("--realtime", action="store_true",
                    help="with --serve, pace the simulation to wall-clock so it looks natural")
    ap.add_argument("--hold", type=int, default=0, metavar="SECONDS",
                    help="with --serve, keep the page alive this long after the run ends")
    args = ap.parse_args()

    res = run(args)
    print()
    for k, v in res.items():
        print("  {:22s} {}".format(k, v))
    print()
    if res["contact_steps"] == 0 and res["reached"]:
        print("  clean run: reached the goal with no contact.")
    elif not res["reached"]:
        print("  did not reach the goal within the time limit "
              "(try --time_limit, --v_max, or --compliance 1.0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
