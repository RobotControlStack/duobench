"""Meshes of the FR3 duo mount on the Franka pedestal, taken from the RCS assets (~/.rcs).

Local copies until rcs-core ships fr3_duo_mount_franka_pedestal and robotiq_d405_mount_black, remove this script
together with assets/scenes/vention_pedestal/assets/duo_mount then. Requires trimesh and pymeshlab.

- fr3_duo_mount (block) and fr3_duo_cover: decimated visual meshes, RCS collision meshes
- fr3_duo_head (= franka_description franka_head_v0_2, replaces the 3D printed RCS head): decimated, also used
  for collisions (convex hull)
- fr3_duo_pillar_28cm (item profile 8 120x80 light): stretched to the measured 250 mm
- robotiq_d405_mount (wrist camera mount): copied

Writes binary STLs to assets/scenes/vention_pedestal/assets/duo_mount.

Usage: python make_duo_mount_assets.py [<rcs_prefix>]
"""

import os
import shutil
import sys

import convert_vention_step as cvs
import trimesh

PILLAR_LENGTH = 0.25
RCS_PILLAR_LENGTH = 0.28
# fraction of the triangles kept for the visual meshes
KEEP = {"fr3_duo_mount": 0.1, "fr3_duo_head": 0.1, "fr3_duo_cover": 0.25}


OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "assets",
    "scenes",
    "vention_pedestal",
    "assets",
    "duo_mount",
)


def main(out_dir, rcs_prefix):
    src = os.path.join(rcs_prefix, "assets", "objects", "fr3_duo_mount", "assets")
    os.makedirs(out_dir, exist_ok=True)
    for name, keep in KEEP.items():
        m = trimesh.load(os.path.join(src, name + ".obj"), force="mesh")
        v, f = cvs.decimate(m.vertices, m.faces, keep)
        out = trimesh.Trimesh(v, f)
        out.export(os.path.join(out_dir, name + ".stl"))
        print(f"{name}: {len(m.faces)} -> {len(out.faces)} triangles")
        if os.path.exists(os.path.join(src, name + ".stl")):
            shutil.copy(os.path.join(src, name + ".stl"), os.path.join(out_dir, name + "_collision.stl"))
    for ext, suffix in (("obj", ""), ("stl", "_collision")):
        m = trimesh.load(os.path.join(src, f"fr3_duo_pillar_28cm.{ext}"), force="mesh")
        m.apply_scale([1.0, 1.0, PILLAR_LENGTH / RCS_PILLAR_LENGTH])
        m.export(os.path.join(out_dir, f"fr3_duo_pillar_{PILLAR_LENGTH * 1000:.0f}mm{suffix}.stl"))
    shutil.copy(
        os.path.join(rcs_prefix, "assets", "objects", "robotiq_d405_mount", "assets", "robotiq_d405_mount.stl"),
        os.path.join(out_dir, "robotiq_d405_mount.stl"),
    )
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}: {os.path.getsize(os.path.join(out_dir, f)) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(OUT_DIR, sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.rcs"))
