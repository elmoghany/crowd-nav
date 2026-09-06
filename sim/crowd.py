"""Social-force pedestrians.

Each person is pulled toward their goal, pushed away from other people, pushed away from the
robot, and pushed away from walls. `compliance` scales the robot term: 1.0 is a cooperative
crowd that yields, 0.1 is a crowd that essentially ignores the robot.

HONEST NOTE: these are the same force terms the controller's own physics prior uses. A crowd
that behaves exactly like your model is the easy case -- results here are an upper bound on
what you should expect from real people. Lower `compliance` (or write your own model in
`step()`) to make it adversarial.
"""
import numpy as np

DESIRED_SPEED = 1.3      # m/s, comfortable walking pace
SPEED_CAP = 1.8          # m/s
RELAX = 0.5              # s, goal-force relaxation time
A_PED, B_PED, R_PED = 2.0, 0.4, 2.0        # strength, length scale, cutoff radius (m)
A_ROB, B_ROB, R_ROB = 1.0, 0.5, 3.0
A_WALL, B_WALL = 4.0, 0.25


class Crowd:
    def __init__(self, positions, goals, walls=None, compliance=1.0, seed=0, waypoints=None):
        self.p = np.asarray(positions, float).copy()          # (N, 2)
        self.goal = np.asarray(goals, float).copy()           # (N, 2)
        self.v = np.zeros_like(self.p)
        self.walls = [np.asarray(w, float) for w in (walls or [])]   # [cx, cy, hx, hy]
        self.compliance = float(compliance)
        self.rng = np.random.default_rng(seed)
        self.routes = [list(map(np.asarray, wp)) for wp in waypoints] if waypoints else None
        self._leg = [0] * len(self.p)

    @property
    def n(self):
        return len(self.p)

    def _target(self, i):
        """Current waypoint if this pedestrian has a route, else the final goal."""
        if self.routes and self._leg[i] < len(self.routes[i]):
            t = self.routes[i][self._leg[i]]
            if np.linalg.norm(t - self.p[i]) < 0.6:
                self._leg[i] += 1
                return self._target(i)
            return t
        return self.goal[i]

    def step(self, robot_xy, dt):
        """Advance every pedestrian one step. robot_xy: (2,) or None for the ghost (no-robot) run."""
        f = np.zeros_like(self.p)
        for i in range(self.n):
            d = self._target(i) - self.p[i]
            nd = np.linalg.norm(d) + 1e-9
            f[i] += (DESIRED_SPEED * d / nd - self.v[i]) / RELAX          # goal

            other = np.delete(np.arange(self.n), i)
            if len(other):
                rel = self.p[i] - self.p[other]
                dist = np.linalg.norm(rel, axis=1) + 1e-9
                near = dist < R_PED
                if near.any():
                    f[i] += (A_PED * np.exp(-dist[near] / B_PED)[:, None]
                             * (rel[near] / dist[near][:, None])).sum(axis=0)

            if robot_xy is not None and self.compliance > 0:
                rel = self.p[i] - np.asarray(robot_xy, float)
                dist = float(np.linalg.norm(rel)) + 1e-9
                if dist < R_ROB:
                    f[i] += self.compliance * A_ROB * np.exp(-dist / B_ROB) * rel / dist

            for cx, cy, hx, hy in self.walls:                              # keep out of wall boxes
                q = np.clip(self.p[i], [cx - hx, cy - hy], [cx + hx, cy + hy])
                rel = self.p[i] - q
                dist = float(np.linalg.norm(rel))
                if dist < 1e-6:                                            # inside: push out laterally
                    rel = np.array([0.0, 1.0 if self.p[i][1] > cy else -1.0])
                    dist = 1e-6
                if dist < 1.0:
                    f[i] += A_WALL * np.exp(-dist / B_WALL) * rel / dist

        self.v += f * dt
        sp = np.linalg.norm(self.v, axis=1)
        fast = sp > SPEED_CAP
        self.v[fast] *= (SPEED_CAP / sp[fast])[:, None]
        self.p += self.v * dt
        return self.p
