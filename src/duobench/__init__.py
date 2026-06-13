__version__ = "0.1.0"

import os
from pathlib import Path

import numpy as np
import rcs
from rcs import get_prefix
from rcs._core import common

RCS_DUOBENCH_GITHUB_ASSET_ARCHIVE_URL = "https://github.com/RobotControlStack/rcs_duobench/archive/refs/tags/{tag}.zip"

RCS_DUOBENCH_PREFIX = get_prefix(
    os.environ.get("RCS_DUOBENCH_PREFIX"),
    Path(__file__).resolve().parents[2],
    Path.home() / ".rcs_duobench",
    __version__,
    RCS_DUOBENCH_GITHUB_ASSET_ARCHIVE_URL,
)

CAMERA_PATHS: dict[str, str] = {}
GRIPPER_PATHS: dict[str, str] = {}
OBJECT_PATHS: dict[str, str] = {
    # transfer_reorient
    "transfer_reorient_socket": "assets/objects/transfer_reorient/socket.xml",
    "transfer_reorient_hex_peg": "assets/objects/transfer_reorient/hexpeg.xml",
    "transfer_reorient_squere_peg": "assets/objects/transfer_reorient/squere_peg.xml",
    "transfer_reorient_triangel_peg": "assets/objects/transfer_reorient/triangle_peg.xml",
    # block_stacking
    "block_stacking_big_red_cube": "assets/objects/block_balance/big_red_cube.xml",
    "block_stacking_blue_rectangle": "assets/objects/block_balance/blue_rectangle.xml",
    "block_stacking_green_rectangle": "assets/objects/block_balance/green_rectangle.xml",
    "block_stacking_pink_beam": "assets/objects/block_balance/pink_beam.xml",
    # bin_sort
    "parallel_pick_white_bowl": "assets/objects/bin_sort/white_bowl.xml",
    "parallel_pick_white_tri_cylinder": "assets/objects/bin_sort/white_tri_cylinder.xml",
    "parallel_pick_white_box": "assets/objects/bin_sort/white_box.xml",
    "parallel_pick_white_pent_cylinder": "assets/objects/bin_sort/white_pent_cylinder.xml",
    "parallel_pick_white_hex_cylinder": "assets/objects/bin_sort/white_hex_cylinder.xml",
    "parallel_pick_white_oct_cylinder": "assets/objects/bin_sort/white_oct_cylinder.xml",
    "parallel_pick_white_dec_cylinder": "assets/objects/bin_sort/white_dec_cylinder.xml",
    "parallel_pick_white_cylinder": "assets/objects/bin_sort/white_cylinder.xml",
    "parallel_pick_black_bowl": "assets/objects/bin_sort/black_bowl.xml",
    "parallel_pick_black_tri_cylinder": "assets/objects/bin_sort/black_tri_cylinder.xml",
    "parallel_pick_black_box": "assets/objects/bin_sort/black_box.xml",
    "parallel_pick_black_pent_cylinder": "assets/objects/bin_sort/black_pent_cylinder.xml",
    "parallel_pick_black_hex_cylinder": "assets/objects/bin_sort/black_hex_cylinder.xml",
    "parallel_pick_black_oct_cylinder": "assets/objects/bin_sort/black_oct_cylinder.xml",
    "parallel_pick_black_dec_cylinder": "assets/objects/bin_sort/black_dec_cylinder.xml",
    "parallel_pick_black_cylinder": "assets/objects/bin_sort/black_cylinder.xml",
    "parallel_pick_blue_tri_cylinder": "assets/objects/bin_sort/blue_tri_cylinder.xml",
    "parallel_pick_blue_box": "assets/objects/bin_sort/blue_box.xml",
    "parallel_pick_blue_pent_cylinder": "assets/objects/bin_sort/blue_pent_cylinder.xml",
    "parallel_pick_blue_hex_cylinder": "assets/objects/bin_sort/blue_hex_cylinder.xml",
    "parallel_pick_blue_cylinder": "assets/objects/bin_sort/blue_cylinder.xml",
    "parallel_pick_green_tri_cylinder": "assets/objects/bin_sort/green_tri_cylinder.xml",
    "parallel_pick_green_box": "assets/objects/bin_sort/green_box.xml",
    "parallel_pick_green_pent_cylinder": "assets/objects/bin_sort/green_pent_cylinder.xml",
    "parallel_pick_green_hex_cylinder": "assets/objects/bin_sort/green_hex_cylinder.xml",
    "parallel_pick_green_cylinder": "assets/objects/bin_sort/green_cylinder.xml",
    # spring_door
    "spring_door_gray_microwave": "assets/objects/spring_door/gray_microwave.xml",
    "spring_door_mug": "assets/objects/spring_door/mug.xml",
    "spring_door_simple_mug": "assets/objects/spring_door/simple_mug.xml",
    # hinge chest
    "hinge_chest_chest": "assets/objects/hinge_chest/chest.xml",
    # transfer_gate
    "handover_hole_stand": "assets/objects/transfer_gate/stand.xml",
    "handover_hole_mat": "assets/objects/transfer_gate/mat.xml",
    "handover_hole_white_long_box": "assets/objects/transfer_gate/white_long_box.xml",
    # carry pot
    "carry_pot_pot": "assets/objects/carry_pot/pot.xml",
    "carry_pot_stove": "assets/objects/carry_pot/stove.xml",
    # join_blocks
    "h_block": "assets/objects/join_blocks/join_blocks_h_block.xml",
    "p_block": "assets/objects/join_blocks/join_blocks_p_block.xml",
    "wall": "assets/objects/join_blocks/join_blocks_wall.xml",
    # pour_marbles
    "marble": "assets/objects/pour_marbles/marble.xml",
    "teacup": "assets/objects/pour_marbles/teacup.xml",
    # ball_maze
    "maze1": "assets/objects/ball_maze/maze_simple_1.xml",
    "maze2": "assets/objects/ball_maze/maze_simple_2.xml",
    "maze3": "assets/objects/ball_maze/maze_simple_3.xml",
    "maze4": "assets/objects/ball_maze/maze_simple_4.xml",
    "maze5": "assets/objects/ball_maze/maze_simple_5.xml",
    "maze6": "assets/objects/ball_maze/maze_simple_6.xml",
    "maze7": "assets/objects/ball_maze/maze_simple_7.xml",
    "maze8": "assets/objects/ball_maze/maze_simple_8.xml",
    "maze9": "assets/objects/ball_maze/maze_simple_9.xml",
    "maze10": "assets/objects/ball_maze/maze_simple_10.xml",
}


SCENE_PATHS = {
    "empty_world": "assets/scenes/base_world.xml",
    "vention_world": "assets/scenes/vention_table/vention_world.xml",
}

# Append RCS_DUOBENCH package prefix to all asset paths
for path_dict in (GRIPPER_PATHS, SCENE_PATHS, OBJECT_PATHS, CAMERA_PATHS):
    for name, path in path_dict.items():
        abs_path = os.path.join(RCS_DUOBENCH_PREFIX, path)
        if not os.path.isfile(abs_path):
            error_msg = f"Asset {name} not found at path: {abs_path}. Please make sure to download the assets."
            raise FileNotFoundError(error_msg)
        else:
            path_dict[name] = abs_path

# Update the global RCS registries with our assets and tasks
rcs.SCENE_PATHS.update(SCENE_PATHS)
rcs.OBJECT_PATHS.update(OBJECT_PATHS)
rcs.GRIPPER_PATHS.update(GRIPPER_PATHS)
rcs.CAMERA_PATHS.update(CAMERA_PATHS)


rcs.DEFAULT_TRANSFORMS.update(
    {"VENTION_HEIGHT_OFFSET": common.Pose(translation=np.array([0, 0, 0.8095]), quaternion=np.array([0, 0, 0, 1]))}
)
