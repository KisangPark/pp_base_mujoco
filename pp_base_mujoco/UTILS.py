import mujoco
import numpy as np
import time


""" MUJOCO NAMES """
def get_body_names (model, data):
    body_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY, body_idx) for body_idx in range(model.nbody)]
    return body_names

def get_actuator_names (model, data):
    control_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) for ctrl_idx in range(model.nu)]
    return control_names


""" MUJOCO APPLY QPOS """

def get_joint_names (model, data):
    joint_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx) for joint_idx in range(model.njnt)]
    return joint_names

def apply_qpos_idxs (model, data, idxs, value):
    if len(idxs) != len(value):
        raise ValueError("length of name and value is different")
    qpos_ = np.zeros(model.nq) # number of qpos
    for i, idx in enumerate(idxs):
        qpos_[idx] = value[i]
    data.qpos = qpos_    
    return None

def apply_qpos_names (model, data, names, value):
    if len(names) != len(value):
        raise ValueError("length of names and value is different")
    # initialize
    indexs = [model.joint(joint_name).qposadr[0] for joint_name in names]
    qpos_ = np.zeros(model.nq) # number of qpos

    for i, idx in enumerate(indexs):
        qpos_[idx] = value[i]
    data.qpos = qpos_
    return None


""" MUJOCO APPLY CONTROL """

def apply_ctrl_idxs (model, data, idxs, value):
    if len(idxs) != len(value):
        raise ValueError("length of name and value is different")
    ctrl_ = np.zeros(model.nu) # number of control
    for i, idx in enumerate(idxs):
        ctrl_[idx] = value[i]
    data.ctrl = ctrl_    
    return None

def apply_ctrl_names (model, data, names, value):
    if len(names) != len(value):
        raise ValueError("length of names and value is different")
    # initialize
    control_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) for ctrl_idx in range(model.nu)]
    ctrl_ = np.zeros(model.nu) # number of control

    for i, n in enumerate(names):
        if n in control_names:
            idx = control_names.index(n)
            ctrl_[idx] = value[i]
        else:
            print(f"Name {n} is not included in actuator names, passing..")
    data.ctrl = ctrl_
    return None


""" MUJOCO MJ SPEC """

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
    spec = spec = mujoco.MjSpec.from_file(scene_path)

    # for multiple robots, attach frame & body
    for robot_name, robot_config in robots.items():
        # attach frame
        frame_pos = robot_config["frame"]["pos"]
        frame_quat = robot_config["frame"]["quat"]
        frame = spec.worldbody.add_frame(pos=frame_pos, quat=frame_quat)
        # attach body to that frame
        hardware_name = robot_config["body"]["file"]
        hardware_file_path = config[hardware_name]
        spec_robot = mujoco.MjSpec.from_file(hardware_file_path)
        body_name = robot_config["body"]["body_name"]
        frame.attach_body(spec_robot.body(body_name), f'{robot_name}-', '')

    return spec.compile() # returns mujoco model
        