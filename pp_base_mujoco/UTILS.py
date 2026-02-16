import mujoco
import numpy as np

def get_actuator_names (model, data):
    control_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) for ctrl_idx in range(model.nu)]
    return control_names

def apply_control_idx (model, data, idxs, value):
    if len(idxs) != len(value):
        raise ValueError("length of name and value is different")
    ctrl_ = np.zeros(model.nu) # number of control
    for i, idx in enumerate(idxs):
        ctrl_[idx] = value[i]
    data.ctrl = ctrl_    
    return None

def apply_ctrl_name (model, data, name, value):
    if len(name) != len(value):
        raise ValueError("length of name and value is different")
    # initialize
    control_names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) for ctrl_idx in range(model.nu)]
    ctrl_ = np.zeros(model.nu) # number of control

    for i, n in enumerate(name):
        if n in control_names:
            idx = control_names.index(n)
            ctrl_[idx] = value[i]
        else:
            print(f"Name {n} is not included in actuator names, passing..")
    data.ctrl = ctrl_
    return None