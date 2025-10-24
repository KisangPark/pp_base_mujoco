import mujoco as mj
import mujoco_viewer

import numpy as np
import time


def spec_generate_model(config):
    """
    GENERATE MUJOCO MODEL WITH MJSPEC
    - Generate mujoco model from configuration file
    """
    try:
        scene_path = config["scene"]
        robots = config["robots"]
    except:
        raise ValueError("configuration is missing something..")
    spec = spec = mj.MjSpec.from_file(scene_path)

    # for multiple robots, attach frame & body
    for robot_name, robot_config in robots.items():
        # attach frame
        frame_pos = robot_config["frame"]["pos"]
        frame_quat = robot_config["frame"]["quat"]
        frame = spec.worldbody.add_frame(pos=frame_pos, quat=frame_quat)
        # attach body to that frame
        hardware_name = robot_config["body"]["file"]
        hardware_file_path = config[hardware_name]
        spec_robot = mj.MjSpec.from_file(hardware_file_path)
        body_name = robot_config["body"]["body_name"]
        frame.attach_body(spec_robot.body(body_name), f'{robot_name}-', '')

    return spec.compile() # returns mujoco model
        




