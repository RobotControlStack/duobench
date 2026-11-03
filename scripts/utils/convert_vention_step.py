"""Converts the Vention pedestal STEP assembly into the MuJoCo model in assets/scenes/vention_pedestal.

Requires cadquery-ocp (STEP import), trimesh and pymeshlab (decimation).

- drops the robot mounting plates (ST-RB-013-0001)
- converts STEP (Y-up, mm) to MuJoCo (Z-up, m)
- origin: floor level, centered below the top of the four vertical columns
- keeps the CAD colors: each part is split into one visual mesh per face color,
  with one MuJoCo material per color
- writes STLs plus an includable body xml

Variants:
- --simple --merge (used by duobench): coarse tessellation + quadric decimation (meshlab), all visual meshes of
                   one color merged into a single mesh, one box collision per part
                   -> vention_pedestal.xml + assets/cad_*.stl
- --simple:        like above but one visual mesh per part -> vention_pedestal_simple.xml + assets_simple/
- full (default):  fine tessellation (600k triangles), convex-hull mesh collisions
                   -> vention_pedestal_full.xml + assets_full/

Usage: python convert_vention_step.py <assembly.STEP> [--simple [--merge]] [--out DIR]
       (DIR defaults to assets/scenes/vention_pedestal; the pedestal was generated from
       VentionAssembly_525636_v2.13.STEP with --simple --merge)
"""

import os
import sys

import numpy as np
import pymeshlab
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

EXCLUDE = {"ST-RB-013-0001"}  # robot mounting plates
DEFAULT_COLOR = (0.6, 0.6, 0.6)  # faces without a CAD color

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "scenes", "vention_pedestal")

# variant -> output names, tessellation (mm linear / rad angular deflection), decimation
VARIANTS = {
    "full": {"xml": "vention_pedestal_full.xml", "meshdir": "assets_full", "deflection": (0.2, 0.3), "keep": 1.0},
    "simple": {
        "xml": "vention_pedestal_simple.xml",
        "meshdir": "assets_simple",
        "deflection": (1.0, 0.5),
        "keep": 0.15,
    },
    "simple_merged": {
        "xml": "vention_pedestal.xml",
        "meshdir": "assets",
        "deflection": (1.0, 0.5),
        "keep": 0.15,
    },
}
MIN_FACES_TO_DECIMATE = 200


def read_parts(step_path):
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.ReadFile(step_path)
    reader.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    def color(label):
        c = Quantity_Color()
        for ctype in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
            if ct.GetColor_s(label, ctype, c):
                return tuple(round(x, 4) for x in c.Values(Quantity_TOC_sRGB))
        return None

    def face_colors(label):
        """(face, rgb) pairs for faces colored via sub-shape labels."""
        out = []
        subs = TDF_LabelSequence()
        st.GetSubShapes_s(label, subs)
        for i in range(1, subs.Length() + 1):
            rgb = color(subs.Value(i))
            if rgb is None:
                continue
            exp = TopExp_Explorer(st.GetShape_s(subs.Value(i)), TopAbs_FACE)
            is_face = st.GetShape_s(subs.Value(i)).ShapeType() == TopAbs_FACE
            while exp.More():
                out.append((not is_face, exp.Current(), rgb))
                exp.Next()
        # face-level colors override colors of the solid that contains the face
        return [(f, c) for _, f, c in sorted(out, key=lambda x: x[0])]

    def name(label):
        attr = TDataStd_Name()
        return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else "part"

    parts = []

    def walk(label, loc):
        if st.IsAssembly_s(label):
            children = TDF_LabelSequence()
            st.GetComponents_s(label, children)
            for i in range(1, children.Length() + 1):
                comp = children.Value(i)
                ref = TDF_Label()
                st.GetReferredShape_s(comp, ref)
                walk(ref, loc.Multiplied(st.GetLocation_s(comp)))
        else:
            parts.append(
                (name(label), st.GetShape_s(label).Moved(loc), color(label) or DEFAULT_COLOR, face_colors(label))
            )

    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location())
    return parts


def tessellate(shape, part_color, face_colors, deflection):
    """Returns vertices, triangles and the rgb color of each triangle."""
    BRepMesh_IncrementalMesh(shape, deflection[0], False, deflection[1], True)
    verts, faces, colors = [], [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        # IsPartner ignores the instance location, the sub-shape colors are defined on the unplaced part
        rgb = next((c for f, c in face_colors if face.IsPartner(f)), part_color)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            off = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                faces.append(
                    (off + a - 1, off + c - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + c - 1)
                )
                colors.append(rgb)
        exp.Next()
    return np.array(verts), np.array(faces), colors


def decimate(v, f, keep):
    """Quadric edge collapse down to the `keep` fraction of triangles."""
    # OCCT emits separate vertices per CAD face, weld them first or the decimator tears the surface apart
    m = trimesh.Trimesh(v, f, process=True)
    if keep >= 1.0 or len(m.faces) < MIN_FACES_TO_DECIMATE:
        return m.vertices, m.faces
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            np.ascontiguousarray(m.vertices, dtype=np.float64), np.ascontiguousarray(m.faces, dtype=np.int32)
        )
    )
    # preservetopology avoids a segfault in meshlab on the caster mesh
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=keep, preservenormal=True, preservetopology=True, planarquadric=True, qualitythr=0.3, autoclean=True
    )
    out = ms.current_mesh()
    return out.vertex_matrix(), out.face_matrix()


def model_frame(meshes):
    """Rotation and origin (mm) so that model vertices (m) are (v @ rot.T - origin) / 1000.

    STEP Y-up mm -> MuJoCo Z-up m: (x, y, z) -> (x, -z, y); origin on the floor, centered below the columns
    (the vertical extrusions).
    """
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    all_v = np.concatenate([m[1] for m in meshes]) @ rot.T

    def is_column(name, v):
        ext = np.ptp(v @ rot.T, axis=0)
        return name.startswith("ST-EXT") and ext[2] > 2 * ext[:2].max()

    columns = np.concatenate([m[1] for m in meshes if is_column(m[0], m[1])]) @ rot.T
    origin = np.array(
        [
            (columns[:, 0].min() + columns[:, 0].max()) / 2,
            (columns[:, 1].min() + columns[:, 1].max()) / 2,
            all_v[:, 2].min(),
        ]
    )
    return rot, origin


def main(step_path, variant, out_dir=OUT_DIR):
    cfg = VARIANTS[variant]
    simple = variant in ("simple", "simple_merged")
    merge = variant == "simple_merged"
    os.makedirs(out_dir, exist_ok=True)
    mesh_dir = os.path.join(out_dir, cfg["meshdir"])
    parts = [p for p in read_parts(step_path) if p[0] not in EXCLUDE]
    meshes = [(n, *tessellate(s, c, fc, cfg["deflection"])) for n, s, c, fc in parts]

    rot, origin = model_frame(meshes)

    os.makedirs(mesh_dir, exist_ok=True)
    # only remove meshes written by this script, the mesh directory is shared with other scene assets
    for f in os.listdir(mesh_dir):
        if f.endswith(".stl") and f.startswith(("cad_", "st_", "mo_", "hw_")):
            os.remove(os.path.join(mesh_dir, f))

    asset_lines, geom_lines = [], []
    materials = {}  # rgb -> material name
    counts = {}
    top = -np.inf
    n_faces = 0
    merged = {}  # rgb -> list of (vertices, faces), merge variant only

    def add_mesh(name, v, f, extra="", process=True):
        nonlocal n_faces
        m = trimesh.Trimesh(v, f, process=process)
        n_faces += len(m.faces)
        m.export(os.path.join(mesh_dir, name + ".stl"))
        asset_lines.append(f'    <mesh name="{name}" file="{name}.stl"{extra}/>')

    for n, v_step, f, colors in meshes:
        idx = counts.get(n, 0)
        counts[n] = idx + 1
        mesh_name = f"{n.lower().replace('-', '_')}_{idx}"
        v = (v_step @ rot.T - origin) / 1000.0
        top = max(top, v[:, 2].max())
        for rgb in colors:
            materials.setdefault(rgb, "cad_" + "".join(f"{round(x * 255):02x}" for x in rgb))
        unique = list(dict.fromkeys(colors))
        if simple:
            # visual meshes are decimated per color (keeps color borders); decimated meshes are not
            # guaranteed to be closed, shell inertia avoids the closed-volume check
            color_idx = np.array([unique.index(c) for c in colors])
            for k, rgb in enumerate(unique):
                if merge:
                    merged.setdefault(rgb, []).append(decimate(v, f[color_idx == k], cfg["keep"]))
                    continue
                vis_name = mesh_name if len(unique) == 1 else f"{mesh_name}_c{k}"
                add_mesh(vis_name, *decimate(v, f[color_idx == k], cfg["keep"]), ' inertia="shell"')
                geom_lines.append(
                    f'      <geom name="{vis_name}_visual" mesh="{vis_name}" material="{materials[rgb]}" class="visual"/>'
                )
            # all parts are axis aligned in the model frame -> axis aligned bounding box
            lo, hi = v.min(0), v.max(0)
            geom_lines.append(
                f'      <geom name="{mesh_name}_collision" type="box" pos="{" ".join(f"{x:.5f}" for x in (lo + hi) / 2)}"'
                f' size="{" ".join(f"{x:.5f}" for x in (hi - lo) / 2)}" class="collision"/>'
            )
            continue
        # collision uses the convex hull of the whole part
        add_mesh(mesh_name, v, f)
        if len(unique) == 1:
            geom_lines.append(
                f'      <geom name="{mesh_name}_visual" mesh="{mesh_name}" material="{materials[unique[0]]}" class="visual"/>'
            )
        else:
            color_idx = np.array([unique.index(c) for c in colors])
            for k, rgb in enumerate(unique):
                # color patches are open surfaces (visual only), shell inertia avoids the closed-volume check
                add_mesh(f"{mesh_name}_c{k}", v, f[color_idx == k], ' inertia="shell"')
                geom_lines.append(
                    f'      <geom name="{mesh_name}_c{k}_visual" mesh="{mesh_name}_c{k}" material="{materials[rgb]}" class="visual"/>'
                )
        geom_lines.append(f'      <geom name="{mesh_name}_collision" mesh="{mesh_name}" class="collision"/>')
    for rgb, pieces in merged.items():
        name = materials[rgb]
        offsets = np.cumsum([0] + [len(pv) for pv, _ in pieces[:-1]])
        # no re-welding: vertices of different parts must not be merged, otherwise normals get smoothed
        # across part boundaries
        add_mesh(
            name,
            np.concatenate([pv for pv, _ in pieces]),
            np.concatenate([pf + o for (_, pf), o in zip(pieces, offsets, strict=True)]),
            ' inertia="shell"',
            process=False,
        )
        geom_lines.insert(0, f'      <geom name="{name}_visual" mesh="{name}" material="{name}" class="visual"/>')
    material_lines = [
        f'    <material name="{name}" rgba="{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} 1" specular="0.3" shininess="0.3"/>'
        for rgb, name in materials.items()
    ]

    pedestal_xml = f"""<mujoco model="vention_pedestal">
  <!-- generated by scripts/utils/convert_vention_step.py ({variant} variant) from {os.path.basename(step_path)}
       (robot plates removed) -->
  <compiler angle="radian" meshdir="{cfg["meshdir"]}" autolimits="true"/>

  <asset>
{chr(10).join(material_lines)}
{chr(10).join(asset_lines)}
  </asset>

  <default>
    <default class="vention_pedestal">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="vention_pedestal" childclass="vention_pedestal">
{chr(10).join(geom_lines)}
      <!-- center of the pedestal top surface (where the plates were removed) -->
      <site name="pedestal_top" pos="0 0 {top:.4f}" size="0.01" rgba="1 0 0 1" group="4"/>
    </body>
  </worldbody>
</mujoco>
"""
    with open(os.path.join(out_dir, cfg["xml"]), "w") as fh:
        fh.write(pedestal_xml)
    print(
        f"{variant}: wrote {len(meshes)} parts, {n_faces} mesh triangles, {len(materials)} colors"
        f" {list(materials.values())}, top surface at z={top:.4f} m"
    )


if __name__ == "__main__":
    flags = sys.argv[2:]
    out = os.path.abspath(flags[flags.index("--out") + 1]) if "--out" in flags else OUT_DIR
    main(sys.argv[1], ("simple_merged" if "--merge" in flags else "simple") if "--simple" in flags else "full", out)
