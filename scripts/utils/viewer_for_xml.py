import mujoco
import mujoco.viewer
import sys
import os

import numpy as np
import os
import numpy as np
import mujoco


def launch_mujoco_viewer(xml_path: str):
    """
    Loads a MuJoCo model from XML and launches viewer correctly.
    """

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)


    # 🚨 APPLY KEYFRAME (THIS WAS MISSING)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # 🚨 IMPORTANT: propagate state
    mujoco.mj_forward(model, data)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)

    while True:

        mujoco.mj_step(model, data)

        # --- your logic ---
        marker_name = "marker"
        ball_name = "blue_ball"

        marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, marker_name)
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ball_name)

        marker_pos = data.geom_xpos[marker_id]
        ball_pos = data.geom_xpos[ball_id]

        diff = marker_pos - ball_pos
        diff_abs = np.linalg.norm(diff)

        reward = 1 - diff_abs

        #print(diff_abs)

        viewer.sync()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_mujoco.py <path_to_xml_file>")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    launch_mujoco_viewer(xml_file)