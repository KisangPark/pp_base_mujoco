import mujoco
import numpy as np
import time


""" MUJOCO NAMES """
def get_body_names (model, data):
    body_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY, body_idx) for body_idx in range(model.nbody)]
    return body_names

def get_geom_names (model, data):
    geom_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM, geom_idx) for geom_idx in range(model.ngeom)]
    return geom_names

def get_site_names (model, data):
    site_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_SITE, site_idx) for site_idx in range(model.nsite)]
    return site_names

def get_actuator_names (model, data):
    control_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) for ctrl_idx in range(model.nu)]
    return control_names

def get_joint_names (model, data):
    joint_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx) for joint_idx in range(model.njnt)]
    return joint_names

def get_qpos_with_names (model, data, names):
    qpos_ = np.zeros(len(names)) # number of qpos
    for i, name in enumerate(names):
        idx = model.joint(name).qposadr[0]
        qpos_[i] = data.qpos[idx]
    return qpos_

""" MUJOCO APPLY QPOS """
    
def apply_qpos_idxs (model, data, idxs, value):
    if len(idxs) != len(value):
        raise ValueError("length of name and value is different")
    qpos_ = np.zeros(model.nq) # number of qpos
    for i, idx in enumerate(idxs):
        data.qpos[idx] = value[i]

def apply_qpos_names (model, data, names, value):
    if len(names) != len(value):
        raise ValueError("length of names and value is different")
    # initialize
    indexs = [model.joint(joint_name).qposadr[0] for joint_name in names]
    for i, idx in enumerate(indexs):
        data.qpos[idx] = value[i]


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


""" TRANSFORMS """
def euler2rmat(rpy_list):
    roll, pitch, yaw = rpy_list
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    R = R_z @ R_y @ R_x
    return R

def rmat2euler(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2,1] , R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def euler2quat(rpy_list):
    roll, pitch, yaw = rpy_list
    R = euler2rmat(roll, pitch, yaw)
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4 * qw)
    qy = (R[0,2] - R[2,0]) / (4 * qw)
    qz = (R[1,0] - R[0,1]) / (4 * qw)
    return np.array([qx, qy, qz, qw])

def rmat2rotvec(R):
    """
    Converts a 3x3 rotation matrix to a rotation vector (axis * angle).
    This function is required to compute the joint error for rotation with jacobian matrix.
    """
    # 1. Find the angle theta using the trace of the matrix
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # clip to prevent numerical issues
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    # 2. rotation axis: skew-symmetric components
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz])
    axis = axis / (2.0 * np.sin(theta)) # normalize 
    
    return theta * axis