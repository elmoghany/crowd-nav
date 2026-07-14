"""
PeRoI controller — standalone, simulator-free.

Drop this ONE file (plus the weights `residual_predictor.pt`) onto a real robot. It has no MuJoCo /
gym / project dependency — only `torch` and `numpy`. It wraps the PeRoI action-conditioned pedestrian
predictor (a NeuRoSFM residual trained on real robot–human interaction data) in a holonomic
velocity-grid MPC: every replan it enumerates candidate robot velocities, predicts how each nearby
pedestrian will respond to that candidate robot path, and picks the velocity that best trades goal
progress against predicted clearance (and, optionally, hazard-zone avoidance and corridor keep-in).

USAGE (real-robot control loop)
-------------------------------
    from peroi_controller import PeRoIController
    ctrl = PeRoIController("residual_predictor.pt", robot_radius=0.30, ped_radius=0.25, v_max=0.8)
    ctrl.reset()
    # ... your loop, ideally ~4-10 Hz ...
    while running:
        robot_xy   = (rx, ry)                 # robot position, WORLD/map frame, metres
        goal_xy    = (gx, gy)                  # goal position, same frame
        peds       = {tid: (px, py) for tid, (px, py) in tracked_people.items()}   # {track_id: (x,y)}
        vx, vy = ctrl.step(robot_xy, goal_xy, peds, dt=loop_dt)   # desired velocity, WORLD frame (m/s)
        # send (vx, vy) to the base. For a holonomic base, rotate into the base frame; for a
        # differential-drive base, convert to (v, omega) — see README.

Everything is in a single fixed WORLD frame (e.g. your SLAM/`map` frame): robot pose, goal, and the
tracked pedestrian positions must all be expressed in it, in metres. The predictor observes each
pedestrian for PAST=4 samples at DT=0.25 s and predicts FUTURE=8 samples (2 s). `step()` may be called
at any rate ≥ ~4 Hz; it internally samples/replans at 0.25 s and holds the command in between.

SAFETY: this is a social-navigation *planner*, not a safety layer. Keep your platform's own emergency
stop / collision monitor / speed governor in the loop underneath it. Start with a low `v_max`.
"""
import math
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn

PAST = 4
FUTURE = 8
DT = 0.25
FEAT_DIM_NOTREAT = 2 * PAST + 2 + 2 + 2 + 1 + 4      # = 19
FEAT_DIM_TREAT = FEAT_DIM_NOTREAT + 3                # = 22  (robot-condition one-hot appended)


# ============================================================ model + baselines
class ResidualPredictor(nn.Module):
    """ŷ = SFM_baseline + residual(features). Matches the trained `residual_predictor.pt`."""
    def __init__(self, in_dim=FEAT_DIM_TREAT, out_dim=FUTURE * 2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, sfm_baseline):
        return sfm_baseline + self.net(x)


def _sfm_step(p, v, goal, others, robot_pos, k_robot, dt=DT):
    """One Social-Force step: goal attraction + pedestrian repulsion + (optional) robot repulsion."""
    desired_speed = 1.3
    direction = goal - p
    dist = np.linalg.norm(direction) + 1e-6
    f_goal = (desired_speed * direction / dist - v) / 0.5
    f_ped = np.zeros(2)
    for op in others:
        d = p - op
        nd = np.linalg.norm(d) + 1e-6
        if nd < 2.0:
            f_ped += 2.0 * np.exp(-nd / 0.4) * d / nd
    f_robot = np.zeros(2)
    if robot_pos is not None and k_robot > 0:
        d = p - robot_pos
        nd = np.linalg.norm(d) + 1e-6
        if nd < 3.0:
            f_robot = k_robot * np.exp(-nd / 0.5) * d / nd
    v_new = v + (f_goal + f_ped + f_robot) * dt
    sp = np.linalg.norm(v_new)
    if sp > 1.8:
        v_new = v_new * 1.8 / sp
    return p + v_new * dt, v_new


def _predict_sfm(scene, k_robot):
    p = scene["past"][-1].copy()
    v = (scene["past"][-1] - scene["past"][-2]) / DT if scene["past"].shape[0] > 1 else np.zeros(2)
    goal, others = scene["goal"], list(scene["others"])
    out = []
    for t in range(FUTURE):
        rp = scene["robot_future"][t] if k_robot > 0 else None
        p, v = _sfm_step(p, v, goal, others, rp, k_robot)
        out.append(p.copy())
    return np.array(out)


def _featurize(scene):
    """Pedestrian local-frame feature vector (with the moving-robot treatment one-hot). Returns
    (feats[FEAT_DIM_TREAT], current_position[2])."""
    MAX_ROBOT_DIST = 10.0
    past = scene["past"]; cur = past[-1]
    vel = (past[-1] - past[-2]) / DT if past.shape[0] > 1 else np.zeros(2)
    goal_dir = scene["goal"] - cur
    goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
    rel_robot_raw = scene["robot_state"] - cur
    if np.linalg.norm(rel_robot_raw) > MAX_ROBOT_DIST:
        rel_robot, robot_nearby = np.zeros(2, np.float32), 0.0
    else:
        rel_robot, robot_nearby = rel_robot_raw.astype(np.float32), 1.0
    others = scene["others"]
    if len(others) > 0:
        rel = np.asarray(others) - cur
        idx = np.argsort(np.linalg.norm(rel, axis=1))[:2]
        top = rel[idx]
        if len(top) < 2:
            top = np.concatenate([top, np.zeros((2 - len(top), 2))], axis=0)
    else:
        top = np.zeros((2, 2))
    feats = np.concatenate([(past - cur).flatten(), vel, goal_dir, rel_robot,
                            np.array([robot_nearby], np.float32), top.flatten(),
                            np.array([0, 0, 1], np.float32)]).astype(np.float32)   # one-hot = moving_robot
    return feats, cur


def _point_in_poly(p, poly):
    x, y = float(p[0]), float(p[1]); n = len(poly); inside = False; j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def load_predictor(path, device="cpu"):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, ResidualPredictor):
        return obj.to(device).eval()
    m = ResidualPredictor(in_dim=FEAT_DIM_TREAT).to(device)
    sd = obj if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()) \
        else next(v for v in obj.values() if isinstance(v, dict))
    m.load_state_dict(sd)
    return m.eval()


# ============================================================ controller
class PeRoIController:
    """Simulator-free PeRoI MPC controller. See module docstring for the loop contract.

    Args:
        ckpt: path to residual_predictor.pt (the trained model).
        robot_radius, ped_radius: body radii (m); clearance is measured surface-to-surface.
        v_max: max commanded speed (m/s) — keep low for first real-robot tests.
        mode: "residual" (trained predictor, default), "sfm" (physics baseline), or "cv" (constant
              velocity) — same MPC planner, different anticipation model (for ablation/debugging).
        hazards: optional list of polygons [[x,y],...] (world frame) the robot should avoid and should
                 not herd people into; enables the hazard-aware cost.
        half_width, corridor_axis: optional corridor keep-in — if the robot runs in a known corridor,
                 pass its half-width (m) and unit axis (x,y); the planner keeps the robot within it.
    """
    def __init__(self, ckpt=None, device="cpu", robot_radius=0.30, ped_radius=0.25, v_max=0.8,
                 mode="residual", hazards=None, half_width=None, corridor_axis=None,
                 k_near=6, consider_r=4.5, fwd_grid=(0.2, 0.45, 0.7, 0.9),
                 lat_grid=(-0.6, -0.3, 0.0, 0.3, 0.6), w_progress=1.0, w_clear=6.0, safety=0.30,
                 comfort=0.8, w_smooth=0.3, w_wall=8.0, w_hazard=6.0):
        assert mode in ("cv", "sfm", "residual")
        self.mode = mode
        self.model = load_predictor(ckpt, device) if mode == "residual" else None
        if mode == "residual" and self.model is None:
            raise ValueError("mode='residual' needs a ckpt path")
        self.device = device
        self.rr = float(robot_radius) + float(ped_radius)
        self.v_max = float(v_max)
        self.hazards = [np.asarray(h, float) for h in hazards] if hazards else None
        self.half_w = float(half_width) if half_width is not None else None
        self.axis = np.asarray(corridor_axis, float) if corridor_axis is not None else None
        self.k_near, self.consider_r = k_near, consider_r
        self.fwd_grid, self.lat_grid = fwd_grid, lat_grid
        self.w_progress, self.w_clear, self.safety = w_progress, w_clear, safety
        self.comfort, self.w_smooth, self.w_wall, self.w_hazard = comfort, w_smooth, w_wall, w_hazard
        self.reset()

    def reset(self):
        self.buf = defaultdict(lambda: deque(maxlen=PAST))   # track_id -> deque[pos] at 0.25 s
        self._vel = np.zeros(2)
        self._t = 0.0
        self._next = 0.0

    # ---- main entry ----
    def step(self, robot_xy, robot_goal, peds, dt=DT):
        """robot_xy,(robot_)goal: (x,y) world frame. peds: {track_id:(x,y)} or list[(x,y)] world frame.
        dt: seconds since the previous call. Returns (vx, vy) desired WORLD-frame velocity (m/s)."""
        rpos = np.asarray(robot_xy, float)
        goal = np.asarray(robot_goal, float)
        ped_items = peds.items() if isinstance(peds, dict) else enumerate(peds)
        ped_map = {k: np.asarray(v, float) for k, v in ped_items}

        self._t += float(dt)
        if self._t + 1e-9 >= self._next:            # 0.25 s sample tick -> buffer + replan
            self._next = self._t + DT
            seen = set(ped_map)
            for tid, p in ped_map.items():
                b = self.buf[tid]
                if b and np.linalg.norm(p - b[-1]) > 1.5:   # track jump -> restart its history
                    b.clear()
                b.append(p)
            for tid in [t for t in self.buf if t not in seen]:   # drop vanished tracks
                del self.buf[tid]
            self._vel = self._replan(rpos, goal, ped_map)
        return float(self._vel[0]), float(self._vel[1])

    # ---- planner ----
    def _candidates(self, rpos, goal):
        d = goal - rpos; n = np.linalg.norm(d)
        fwd = d / n if n > 1e-6 else np.array([1.0, 0.0])
        perp = np.array([-fwd[1], fwd[0]])
        out = []
        for f in self.fwd_grid:
            for l in self.lat_grid:
                v = f * fwd + l * perp
                s = np.linalg.norm(v)
                out.append(v / s * self.v_max if s > self.v_max else v)
        return out, fwd

    def _nearby(self, rpos, ped_map):
        order = sorted(ped_map, key=lambda t: np.linalg.norm(ped_map[t] - rpos))
        picks = []
        for t in order:
            if np.linalg.norm(ped_map[t] - rpos) > self.consider_r:
                break
            picks.append(t)
            if len(picks) >= self.k_near:
                break
        return picks

    def _predict_futures(self, rpos, rfut, near, rich, ped_map):
        pfuts, used = [], set()
        def scene(tid):
            past = np.array(self.buf[tid], float)
            vel = (past[-1] - past[-2]) / DT
            others = np.array([ped_map[j] for j in near if j != tid]) if len(near) > 1 else np.zeros((0, 2))
            return {"past": past, "goal": past[-1] + vel * 3.0, "robot_state": rpos,
                    "robot_future": rfut, "others": others}
        if self.mode == "residual" and rich:
            fb, sb, cur = [], [], []
            for tid in rich:
                sc = scene(tid); f, c = _featurize(sc); s = _predict_sfm(sc, k_robot=1.0)
                fb.append(f); sb.append((s - c).flatten()); cur.append(c)
            with torch.no_grad():
                pred = self.model(torch.tensor(np.array(fb), dtype=torch.float32),
                                  torch.tensor(np.array(sb), dtype=torch.float32)).numpy()
            for j, tid in enumerate(rich):
                pfuts.append(cur[j] + pred[j].reshape(FUTURE, 2)); used.add(tid)
        elif self.mode == "sfm":
            for tid in rich:
                pfuts.append(_predict_sfm(scene(tid), k_robot=1.0)); used.add(tid)
        for tid in near:                                  # history-poor / cv-mode -> constant velocity
            if tid in used:
                continue
            b = self.buf[tid]
            vel = (b[-1] - b[-2]) / DT if len(b) >= 2 else np.zeros(2)
            pfuts.append(np.array([ped_map[tid] + vel * DT * (k + 1) for k in range(FUTURE)]))
        return pfuts

    def _replan(self, rpos, goal, ped_map):
        near = self._nearby(rpos, ped_map)
        rich = [t for t in near if len(self.buf[t]) >= PAST]
        cands, fwd = self._candidates(rpos, goal)
        best_v, best_cost, best_clear = self._vel, 1e18, -1e18
        for v in cands:
            rfut = np.array([rpos + v * DT * (k + 1) for k in range(FUTURE)])
            pfuts = self._predict_futures(rpos, rfut, near, rich, ped_map)
            clear = 1e18
            for pf in pfuts:
                clear = min(clear, float(np.min(np.linalg.norm(rfut - pf, axis=1))) - self.rr)
            progress = float((rfut[-1] - rpos) @ fwd)
            cost = -self.w_progress * progress + self.w_smooth * float(np.linalg.norm(v - self._vel))
            if clear < self.comfort:
                cost += self.w_clear * (self.comfort - max(clear, -0.2))
            if self.half_w is not None and self.axis is not None:   # optional corridor keep-in
                perp = np.array([-self.axis[1], self.axis[0]])
                excess = max(0.0, abs(float(rfut[-1] @ perp)) - (self.half_w - self.rr))
                cost += self.w_wall * excess
            if self.hazards:
                cost += self.w_hazard * self._hazard_penalty(rfut, pfuts)
            feasible = clear >= self.safety
            if best_clear < self.safety:
                if feasible or clear > best_clear:
                    best_v, best_cost, best_clear = v, cost, clear
            elif feasible and cost < best_cost:
                best_v, best_cost, best_clear = v, cost, clear
        return best_v

    def _hazard_penalty(self, rfut, pfuts):
        pen = 0.0
        for hp in rfut:
            if any(_point_in_poly(hp, hz) for hz in self.hazards):
                pen += 1.0
        for pf in pfuts:
            for hp in pf:
                if any(_point_in_poly(hp, hz) for hz in self.hazards):
                    pen += 1.0
                    break
        return pen
