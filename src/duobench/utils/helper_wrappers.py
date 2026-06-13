from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import rcs
from scipy.spatial.transform import Rotation as R


def check_name_occurence(contact_toubles, name, count):
    # Flatten the list of tuples into a single list
    all_names = [item for tup in contact_toubles for item in tup]

    # Count the occurrences of the name in the flattened list
    return all_names.count(name) == count


def get_geom_pos(model, data, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id == -1:
        msg = f"Geom '{name}' not found"
        raise ValueError(msg)
    return data.geom_xpos[geom_id]


def get_body_euler_xyz(model, data, name, degrees=True):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        msg = f"Body '{name}' not found"
        raise ValueError(msg)

    quat = data.xquat[body_id]  # [w, x, y, z]

    # Convert to scipy format [x, y, z, w]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

    return r.as_euler("xyz", degrees=degrees)  # [roll (X), pitch (Y), yaw (Z)]


def invert_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Inverts a homogeneous transformation matrix."""
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    matrix_inv = np.eye(4)
    matrix_inv[:3, :3] = R_inv
    matrix_inv[:3, 3] = t_inv
    return matrix_inv


def get_site_T(data, site_id):
    w_goal_site_pos = data.site_xpos[site_id]
    w_goal_site_mat = data.site_xmat[site_id]
    w_goal_site_T = np.eye(4)
    w_goal_site_T[:3, :3] = w_goal_site_mat.reshape(3, 3)
    w_goal_site_T[:3, 3] = w_goal_site_pos
    return w_goal_site_T


def get_body_T(data, body_id):
    w_target_body_pos = data.xpos[body_id]
    w_target_body_mat = data.xmat[body_id]
    w_target_body_T = np.eye(4)
    w_target_body_T[:3, :3] = w_target_body_mat.reshape(3, 3)
    w_target_body_T[:3, 3] = w_target_body_pos
    return w_target_body_T


def body_pose_in_site_frame(data, body_id, site_id):
    w_goal_site_T = get_site_T(data, site_id)
    w_target_body_T = get_body_T(data, body_id)
    return invert_homogeneous(w_goal_site_T) @ w_target_body_T
    # gs_target_body_pos = gs_target_body_T[:3, 3]


def get_bodies_in_contact_with_gripper_pad(
    sim, arm_side: str, gripper_pad_body_pattern: str = "gripper<arm>_<finger>_pad"
):
    gripper_pad_left_name = gripper_pad_body_pattern.replace("<arm>", arm_side).replace("<finger>", "left")
    gripper_pad_right_name = gripper_pad_body_pattern.replace("<arm>", arm_side).replace("<finger>", "right")
    gripper_pad_left_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, gripper_pad_left_name)
    gripper_pad_right_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, gripper_pad_right_name)

    if gripper_pad_left_id == -1:
        msg = f"Body '{gripper_pad_left_name}' not found"
        raise ValueError(msg)
    if gripper_pad_right_id == -1:
        msg = f"Body '{gripper_pad_right_name}' not found"
        raise ValueError(msg)

    left_contacts: set[int] = set()
    right_contacts: set[int] = set()
    excluded_body_ids = {gripper_pad_left_id, gripper_pad_right_id}

    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        geom1_body_id = sim.model.geom_bodyid[contact.geom1]
        geom2_body_id = sim.model.geom_bodyid[contact.geom2]

        if geom1_body_id == gripper_pad_left_id and geom2_body_id not in excluded_body_ids:
            left_contacts.add(geom2_body_id)
        elif geom2_body_id == gripper_pad_left_id and geom1_body_id not in excluded_body_ids:
            left_contacts.add(geom1_body_id)

        if geom1_body_id == gripper_pad_right_id and geom2_body_id not in excluded_body_ids:
            right_contacts.add(geom2_body_id)
        elif geom2_body_id == gripper_pad_right_id and geom1_body_id not in excluded_body_ids:
            right_contacts.add(geom1_body_id)

    return sorted(left_contacts & right_contacts)


def check_contact_between_bodies(sim, body_id_1, body_id_2):
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        geom1_body_id = sim.model.geom_bodyid[contact.geom1]
        geom2_body_id = sim.model.geom_bodyid[contact.geom2]

        if (geom1_body_id == body_id_1 and geom2_body_id == body_id_2) or (
            geom1_body_id == body_id_2 and geom2_body_id == body_id_1
        ):
            return True
    return False


class RandomSquareObjsPos(gym.Wrapper):
    """
    Wrapper to position arbitrary number of objects in a simulated environment in random spots inside a defined square.
    """

    def __init__(
        self,
        env: gym.Env,
        center2world: rcs.common.Pose,
        obj_joint_names: list[str] | None = None,
        avoid_joint_names: list[str] | None = None,
        obj_position_margin: float = 0.05,
        avoid_position_margin: float | None = None,
        x_width: float = 0.3,
        y_width: float = 0.3,
        z_init: float = 0.02,
        include_rotation: bool = True,
        flip_x: bool = False,
    ):
        super().__init__(env)
        self.obj_joint_names = ["box_joint"] if obj_joint_names is None else obj_joint_names
        self.include_z_rotation = include_rotation
        self.flip_x = flip_x
        self.center2world = center2world
        self.x_width = x_width
        self.y_width = y_width
        self.obj_position_margin = obj_position_margin
        self.avoid_joint_names = [] if avoid_joint_names is None else avoid_joint_names
        self.avoid_position_margin = obj_position_margin if avoid_position_margin is None else avoid_position_margin
        self.z_init = z_init

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        def get_random_position(x_width, y_width):
            return np.array(
                [
                    self.np_random.uniform(-x_width / 2, x_width / 2),
                    self.np_random.uniform(-y_width / 2, y_width / 2),
                    self.z_init,
                ]
            )

        def candidate_is_valid(candidate_pos: np.ndarray, other_positions: list[np.ndarray]) -> bool:
            if other_positions:
                pos_norms = [float(np.linalg.norm(candidate_pos - other_pos)) for other_pos in other_positions]
                if min(pos_norms) < self.obj_position_margin:
                    return False

            if self.avoid_joint_names:
                candidate_world_pos = (self.center2world * rcs.common.Pose(translation=candidate_pos)).translation()
                sim = self.get_wrapper_attr("sim")
                for joint_name in self.avoid_joint_names:
                    avoided_world_pos = np.asarray(sim.data.joint(joint_name).qpos[:3], dtype=float)
                    if np.linalg.norm(candidate_world_pos[:2] - avoided_world_pos[:2]) < self.avoid_position_margin:
                        return False

            return True

        # place randomly in square with sufficient margin between objects
        obj_positions = [get_random_position(self.x_width, self.y_width) for _ in self.obj_joint_names]
        for i, obj_joint_name in enumerate(self.obj_joint_names):
            other_positions = [obj_positions[j] for j in range(len(self.obj_joint_names)) if j != i]
            while not candidate_is_valid(obj_positions[i], other_positions):
                obj_positions[i] = get_random_position(self.x_width, self.y_width)

            theta = self.np_random.uniform(0, 2 * np.pi) if self.include_z_rotation else 0
            x = np.pi / 2 if self.flip_x and self.np_random.random() < 0.5 else 0

            pose_in_center_frame = rcs.common.Pose(translation=obj_positions[i], rpy_vector=np.array([x, 0, theta]))
            pose_in_world_frame = self.center2world * pose_in_center_frame

            # qpos array format for a free joint: [x, y, z, qw, qx, qy, qz]
            self.get_wrapper_attr("sim").data.joint(obj_joint_name).qpos = np.append(
                pose_in_world_frame.translation(), pose_in_world_frame.rotation_q_wxyz()
            )
        # reset of remaining stack, must happen after our reset!
        return super().reset(seed=seed, options=options)
