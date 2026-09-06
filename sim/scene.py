"""Build a MuJoCo scene (MJCF) for a scenario: ground, walls, the Go2, and N pedestrians.

The robot and the pedestrians are placed kinematically every frame (see sim/simulate.py), so
this file only has to describe what things look like and where they start.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))

WALL_H = 0.55      # half-height. Waist-high so the camera can see over the walls; the crowd
                   # model treats a wall as a full barrier however tall it is drawn.


def _xml_path(p):
    return os.path.abspath(p).replace("\\", "/")      # MuJoCo wants forward slashes on Windows


def scene_dir():
    """Generated scenes must live beside go2.xml: the vendored model declares a RELATIVE
    meshdir ("assets"), which MuJoCo resolves against the including file's directory."""
    return os.path.join(ASSETS, "unitree_go2")


PED_BODY = """    <body name="ped{i}" pos="{x} {y} 0">
      <geom class="ped" fromto="0 0 0.05 0 0 0.48" size="0.09" material="ped_legs"/>
      <geom class="ped" fromto="0 0 0.48 0 0 1.42" size="0.17" material="ped_shirt"/>
      <geom class="ped" type="sphere" pos="0 0 1.58" size="0.12" material="ped_skin"/>
    </body>
"""

WALL_GEOM = ('    <geom name="wall{i}" type="box" pos="{cx} {cy} {h}" size="{hx} {hy} {h}" '
             'rgba="0.30 0.33 0.38 1" contype="4" conaffinity="0"/>\n')


def build(scenario, out_path=None):
    """Write the scene XML for `scenario` (a dict, see sim/scenarios/*.yaml) and return its path.

    Only the file NAME of out_path is honoured -- the directory is fixed by scene_dir()."""
    name = os.path.basename(out_path) if out_path else "_scene_generated.xml"
    out_path = os.path.join(scene_dir(), name)

    go2 = "go2.xml"          # sits beside this generated file (see scene_dir)
    ped = _xml_path(os.path.join(ASSETS, "pedestrian", "pedestrian.xml"))

    peds = "".join(PED_BODY.format(i=i, x=p["pos"][0], y=p["pos"][1])
                   for i, p in enumerate(scenario["pedestrians"]))
    walls = "".join(WALL_GEOM.format(i=i, cx=w[0], cy=w[1], hx=w[2], hy=w[3], h=WALL_H)
                    for i, w in enumerate(scenario.get("walls", [])))
    rx, ry = scenario["robot"]["start"]
    gx, gy = scenario["robot"]["goal"]

    xml = """<mujoco model="crowd_nav_scene">
  <include file="{go2}"/>
  <include file="{ped}"/>

  <statistic center="0 0 1.0" extent="8"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048"/>
    <map force="0.01"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".52 .56 .60" rgb2=".46 .50 .55"
             width="512" height="512" mark="none"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="goalmat" rgba="0.15 0.65 0.35 0.7"/>
  </asset>

  <worldbody>
    <light pos="0 0 9" dir="0 0 -1" directional="true" diffuse="0.45 0.45 0.45"
           specular="0.05 0.05 0.05" ambient="0.28 0.28 0.30"/>
    <geom name="floor" type="plane" size="150 80 0.05" material="grid"
          contype="4" conaffinity="0"/>
    <geom name="goal" type="cylinder" pos="{gx} {gy} 0.01" size="0.5 0.01" material="goalmat"
          contype="0" conaffinity="0"/>
{walls}{peds}  </worldbody>
</mujoco>
""".format(go2=go2, ped=ped, rx=rx, ry=ry, gx=gx, gy=gy, walls=walls, peds=peds)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(xml)
    return out_path
