# Running the simulator from Windows

Four ways, depending on where you want the compute to happen. Every command below was run and
verified; where something does **not** work, that is stated rather than left for you to discover.

| you want | use | works? |
|---|---|---|
| simplest thing that works | **A — run it natively on Windows** | ✅ |
| heavy runs on a Linux server, watched live on Windows | **B — `--serve` + SSH tunnel** | ✅ verified end to end |
| heavy runs, just want the result | **C — render to mp4, copy it back** | ✅ |
| Isaac Sim specifically | **D — run Isaac on Windows locally** | ✅ (streaming Isaac *from* the cluster does not work — see below) |

---

## A. Native on Windows (start here)

The simulator is pure pip — no WSL, no Docker, no GPU. This is how it was developed.

```powershell
git clone https://github.com/elmoghany/crowd-nav
cd crowd-nav
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-sim.txt
python -m sim.simulate --scenario doorway --video doorway.mp4
```

Rendering works out of the box on Windows — no `MUJOCO_GL` needed. A 12-pedestrian doorway run
takes about a minute.

---

## B. Compute on Linux, watch live on Windows

Use this when you want many pedestrians, long episodes, or a sweep, but still want to *see* it.

The simulator can stream itself to a browser over one TCP port:

```bash
python -m sim.simulate --scenario doorway --serve 8791 --realtime --hold 300
```

```
live view:
    http://localhost:8791
    http://joachims-cpu-02:8791
    http://128.84.96.55:8791
```

- `--serve PORT` streams MJPEG (plain HTTP, one TCP port, no plugins, no UDP).
- `--realtime` paces the run to wall-clock so it looks natural instead of finishing instantly.
- `--hold N` keeps the page alive N seconds after the run ends, so you can open it late.

### On a cluster, you need an SSH tunnel

**High ports are firewalled** on the Cornell cluster — the address the program prints is not
reachable from your laptop even on the VPN (verified: port 22 open, port 8791 refused). Port 22
*is* open, so tunnel through it. From Windows PowerShell:

```powershell
# 1. find which node your job is on
ssh me484@unicorn-login-01.coecis.cornell.edu "squeue -u me484 -o '%.18i %.20j %.20N'"

# 2. tunnel that node's port to your laptop (leave this window open)
ssh -N -L 8791:joachims-cpu-02:8791 me484@unicorn-login-01.coecis.cornell.edu

# 3. open http://localhost:8791 in your browser
```

The `-L localport:node:remoteport` form hops through the login node to the compute node, so it
works even though the compute node is not directly reachable.

Full recipe as a batch job:

```bash
cat > serve.sbatch <<'EOS'
#!/bin/bash
#SBATCH --job-name=simserve
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=4 --mem=8G --time=00:20:00
#SBATCH --output=serve_%j.log
cd $HOME/crowd-nav
export MUJOCO_GL=egl
source .venv/bin/activate
echo "NODE=$(hostname)"
python -m sim.simulate --scenario doorway --serve 8791 --realtime --hold 600
EOS
sbatch serve.sbatch
grep NODE serve_*.log        # then tunnel to that node
```

---

## C. Render on Linux, copy the video back

Least moving parts. Nothing to keep open, nothing to tunnel.

```bash
# on the Linux box
export MUJOCO_GL=egl
python -m sim.simulate --scenario doorway --video doorway.mp4
```

```powershell
# on Windows
scp me484@unicorn-login-01.coecis.cornell.edu:~/crowd-nav/doorway.mp4 .
```

---

## D. Isaac Sim

**Run Isaac on Windows, locally.** Isaac Sim ships a Windows build, and the research repo's Isaac
environment is plain Python. Install Isaac Sim for Windows, then:

```powershell
git clone https://github.com/elmoghany/crowd-nav-legacy
cd crowd-nav-legacy
# use Isaac's own Python
& "$env:LOCALAPPDATA\ov\pkg\isaac-sim-*\python.bat" isaac_env\run_isaac_episode.py `
    --scenario np1_narrow_passage --method foresight --crowd ours `
    --render --overlay --out out.mp4
```

**Streaming Isaac *from* the cluster does not work, and it is worth knowing why.** Isaac's
livestream extension is installed there (`omni.kit.livestream.webrtc`), but WebRTC carries media
over **UDP**, and an SSH tunnel forwards **TCP only**. Combined with the high-port firewall above,
there is no path from a cluster compute node to your laptop for it. Options that do work:

1. Run Isaac on Windows locally (above) — best if your GPU can take it.
2. Render to mp4 on the cluster and copy it back (option C) — this is what the research pipeline
   does, on the RTX nodes with driver ≥ 580.
3. Use this package's MuJoCo simulator for interactive work (option B) and save Isaac for the
   photoreal renders where it earns its cost.

---

## Troubleshooting

| symptom | cause |
|---|---|
| Browser shows nothing, page never loads | the tunnel is not up, or you tunneled to the login node instead of the compute node. `-L 8791:<compute-node>:8791` |
| Page loads, image stays grey | the run already finished. Use `--hold 300`, or reload |
| `channel ... open failed: administratively prohibited` | that cluster blocks forwarding to that host/port; check the node name is right |
| Stream is choppy | expected — MJPEG re-sends whole frames. Lower `--width 640 --height 360` |
| Works on Windows, black video on Linux | `export MUJOCO_GL=egl` before running |
