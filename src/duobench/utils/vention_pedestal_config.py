import os
import sys
import time

import numpy as np
from rcs import common
from rcs.envs.scenes import SimEnvCreatorConfig

from duobench import DUOBENCH_PREFIX
from duobench.utils.vention_config import VentionSceneFR3Duo

# not part of the downloaded asset archive yet, therefore resolved here instead of registering in SCENE_PATHS
# (which checks all paths on import); DUOBENCH_PREFIX has to contain these assets (an editable install uses the checkout)
ASSET_DIR = os.path.join(DUOBENCH_PREFIX, "assets/scenes/vention_pedestal")

# Franka FR3 Duo pedestal (installation guide R04240): Vention base, mounting plate, 250 mm pillar, duo mount;
# the pillar axis is centered on the pedestal top, 0.0369 m behind the FR3 Duo origin
# TODO: the mount is also added to RCS (fr3_duo_mount_franka_pedestal, robotiq_d405_mount_black,
# EmptyWorldFR3DuoFrankaPedestal); once rcs-core is released with it, use its objects and transforms and remove the
# local copies (fr3_duo_mount_franka_pedestal.xml, fr3_duo_mounting_plate.xml, robotiq_d405_mount_black.xml,
# assets/duo_mount)
PEDESTAL_TOP_TO_ROOT = common.Pose(translation=np.array([0.0369, 0.0, 0.7908]))
MOUNTING_PLATE = (
    os.path.join(ASSET_DIR, "fr3_duo_mounting_plate.xml"),
    common.Pose(translation=np.array([-0.0369, 0, 0])),
)
DUO_MOUNT = (
    os.path.join(ASSET_DIR, "fr3_duo_mount_franka_pedestal.xml"),
    common.Pose(translation=np.array([0, 0, 0.012])),
)
# FR3 Duo origin: 12 mm plate + 250 mm pillar + 50 mm
DUO_ORIGIN = common.Pose(translation=np.array([0.0, 0.0, 0.312]))
# ZED Mini screwed onto the Franka head bracket (holes aligned, left lens at y = 0), 41 deg down
HEAD_CAMERA = common.Pose(translation=np.array([0.00825, -0.02451, 0.66781]), rpy_vector=np.array([0.0, 0.7156, 0.0]))
WRIST_CAMERA_MOUNT = os.path.join(ASSET_DIR, "robotiq_d405_mount_black.xml")

# desk from robot-control-stack master (assets/objects/table), 160 x 80 cm, with the single arm wood texture;
# origin on the floor below the center of its back edge, placed 1 cm in front of the pedestal columns
TABLE_POSE = common.Pose(translation=np.array([0.151, 0.0, 0.0]))
TABLE_HEIGHT = 0.75


def table_xml(height: float) -> str:
    """Desk with the given top height (the legs are adapted)."""
    dz = height - 0.75
    leg = f'size="0.03 0.04 {0.36 + dz / 2:.5f}" pos="0.4 {{}} {0.365 + dz / 2:.5f}"'
    return f"""<mujoco model="table">
  <compiler angle="radian" autolimits="true"/>
  <asset>
    <texture name="wood" type="2d" file="{os.path.join(ASSET_DIR, "assets", "table_wood.png")}"/>
    <material name="wood" texture="wood" specular="0.1" shininess="0.3"/>
    <material name="frame" rgba="0.15 0.15 0.16 1" specular="0.3" shininess="0.4"/>
    <mesh name="top" file="{os.path.join(ASSET_DIR, "assets", "table_top.obj")}" inertia="shell"/>
  </asset>
  <default>
    <default class="visual"><geom type="box" contype="0" conaffinity="0" group="2" material="frame"/></default>
    <default class="collision"><geom type="box" group="3"/></default>
  </default>
  <worldbody>
    <body name="base">
      <geom type="mesh" mesh="top" pos="0.4 0 {0.7375 + dz:.5f}" material="wood" class="visual"/>
      <geom size="0.4 0.8 0.0125" pos="0.4 0 {0.7375 + dz:.5f}" class="collision"/>
      <geom {leg.format(0.7)} class="visual"/>
      <geom {leg.format(-0.7)} class="visual"/>
      <geom {leg.format(0.7)} class="collision"/>
      <geom {leg.format(-0.7)} class="collision"/>
      <geom size="0.35 0.035 0.01" pos="0.4 0.7 0.01" class="visual"/>
      <geom size="0.35 0.035 0.01" pos="0.4 -0.7 0.01" class="visual"/>
      <geom size="0.3 0.03 0.01" pos="0.4 0.7 {0.715 + dz:.5f}" class="visual"/>
      <geom size="0.3 0.03 0.01" pos="0.4 -0.7 {0.715 + dz:.5f}" class="visual"/>
      <geom size="0.02 0.67 0.02" pos="0.4 0 {0.705 + dz:.5f}" class="visual"/>
    </body>
  </worldbody>
</mujoco>
"""


class VentionPedestalFR3Duo(VentionSceneFR3Duo):
    """FR3 duo on the Franka FR3 Duo pedestal (Vention base, mounting plate, pillar, duo mount, Franka head).

    Derives from VentionSceneFR3Duo for its simulation settings (MuJoCo arena size), scene and mount are replaced."""

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        cfg.scene = os.path.join(ASSET_DIR, "vention_pedestal_world.xml")
        cfg.root_frame_to_world = PEDESTAL_TOP_TO_ROOT
        cfg.root_frame_objects = {"mounting_plate": MOUNTING_PLATE, "duo_mount": DUO_MOUNT}
        cfg.shared_base_frame_to_root_frame = DUO_ORIGIN
        cfg.camera_adds["head"].offset = HEAD_CAMERA
        # black wrist camera mounts, same poses as in EmptyWorldFR3Duo
        cfg.robot_frame_objects = {
            robot: {name: (WRIST_CAMERA_MOUNT, pose) for name, (_, pose) in objects.items()}
            for robot, objects in cfg.robot_frame_objects.items()
        }
        return cfg


class VentionPedestalTableFR3Duo(VentionPedestalFR3Duo):
    """VentionPedestalFR3Duo with a table in front of the pedestal."""

    def __init__(self, table_height: float = TABLE_HEIGHT):
        self.table_height = table_height

    def config(self) -> SimEnvCreatorConfig:
        cfg = super().config()
        # RCS loads objects from files, the table is written to the cache for its height
        path = os.path.join(os.path.expanduser("~"), ".cache", "duobench", f"table_{self.table_height * 1000:.0f}.xml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(table_xml(self.table_height))
        cfg.world_frame_objects = {"table": (path, TABLE_POSE)}
        return cfg


if __name__ == "__main__":
    # python -m duobench.utils.vention_pedestal_config [--table]
    scene = VentionPedestalTableFR3Duo() if "--table" in sys.argv[1:] else VentionPedestalFR3Duo()
    env = scene.create_env(scene.config())
    obs, info = env.reset()
    # hold the home pose so the viewer stays open
    act = {
        "left": {"tquat": [0, 0, 0, 0, 0, 0, 1], "gripper": [1]},
        "right": {"tquat": [0, 0, 0, 0, 0, 0, 1], "gripper": [1]},
    }
    while True:
        env.step(act)
        time.sleep(0.033)
