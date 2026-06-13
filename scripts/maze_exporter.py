from pathlib import Path

import mujoco


# IMPORTANT: before using this script remove the ball, the keyframe and the goal and start plates
# then run it, import it in blender and export in stl, this allows to print the respective maze in one piece
# requires pip install 'mujoco[usd]==3.2.6' to be installed

ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / ".." / "assets/objects/ball_maze/maze_simple_3.xml"
OUTPUT_DIR = ROOT / ".." / "usd_export"


def _make_exporter(model: mujoco.MjModel):
    """Create a USD exporter compatible with multiple MuJoCo Python APIs."""
    try:
        from mujoco.usd import exporter as usd_exporter

        return usd_exporter.USDExporter(
            model=model,
            output_directory=OUTPUT_DIR.name,
            output_directory_root=str(ROOT),
        )
    except (ImportError, AttributeError):
        from mujoco import usd

        return usd.USDExporter(model)


def _save_export(exporter_instance) -> Path:
    """Save using whichever exporter method exists in the installed MuJoCo version."""
    if hasattr(exporter_instance, "save_scene"):
        exporter_instance.save_scene(filetype="usd")
        return OUTPUT_DIR

    if hasattr(exporter_instance, "save"):
        output_file = ROOT / "assembled_maze.usd"
        exporter_instance.save(str(output_file))
        return output_file

    msg = "Unsupported MuJoCo USD exporter API: missing both save_scene() and save()."
    raise RuntimeError(msg)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco.mj_forward(model, data)

    exporter_instance = _make_exporter(model)
    exporter_instance.update_scene(data=data)
    output_path = _save_export(exporter_instance)
    print(f"USD export completed: {output_path}")


if __name__ == "__main__":
    main()
