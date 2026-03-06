import os
import sys
import numpy as np
import time
import mujoco

sys.path.append(os.path.dirname(__file__))
from UTILS import * 

""" PREDEFINED CONSTANTS """
POS_ROT_RATIO = 0.57
POSITION_CLIPPING = 0.02
ROTATION_CLIPPING = 0.2

""" JACOBIAN FUNCTIONS """

def get_jacobian(
        model,
        data,
        name,
        type="body",
):
    """
    Get Jacobian of given body/geom/site
    Args
        - model: mujoco model
        - data: mujoco data
        - name: name of the body/geom/site
        - type: "body", "geom", or "site"
    """
    Jacobian_p = np.zeros((3, model.nu))
    Jacobian_r = np.zeros((3, model.nu))
    if type == "body":
        mujoco.mj_jacBody(model, data, Jacobian_p, Jacobian_r, data.body(name).id)
    elif type == "geom":
        mujoco.mj_jacGeom(model, data, Jacobian_p, Jacobian_r, data.geom(name).id)
    elif type == "site":
        mujoco.mj_jacSite(model, data, Jacobian_p, Jacobian_r, data.site(name).id)
    
    return Jacobian_p, Jacobian_r


def get_pseudo_inverse(
        jacobian, # stacked jacobian
        method='svd',
        sigma_threshold=1e-3, # for SVD
        damping=1.0 # for DLS
        ):
    """ 
    Get pseudo-inverse of matrix with SVD or DLS method
    Args:
        - jacobian: Jacobian matrix to be inverted
        - method: method to compute pseudo-inverse, either 'svd' or 'DLS'
        - sigma_threshold: threshold for singular values in SVD method
        - damping: damping factor for DLS method
    """
    row, col = jacobian.shape
    print(f"Jacobian shape: {jacobian.shape}")

    if method=='svd':
        U, Sigma, V_T = np.linalg.svd(jacobian, compute_uv=True)
        print("U shape:", U.shape)
        print("Sigma shape:", Sigma.shape)
        print("V shape:", V_T.shape)

        # suppress singularities with modified sigma
        Sigma_clipped_rev = np.zeros_like(Sigma)
        for i, value in enumerate(Sigma):
            if Sigma[i] < sigma_threshold:
                Sigma_clipped_rev[i] = 0
            else:
                Sigma_clipped_rev[i] = 1/Sigma[i]

        # inverse matrix for position jacobian
        S_rev_matrix = np.zeros((col, row))
        for i, value in enumerate(Sigma_clipped_rev):
            S_rev_matrix[i,i] = value
        inversed = V_T.T @ S_rev_matrix @ U.T

    elif method=='dls':
        # apply damped least squares
        inversed = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping**2 * np.eye(row))
    
    else:
        raise ValueError("Invalid method for pseudo-inverse. Choose either 'svd' or 'dls'.")
    
    return inversed

def get_ik_error_clipped(
        p_current,
        r_current,
        p_target,
        r_target,
        ):
    """
    Get Inverse Kinematics error 
    """
    pos_error = p_target - p_current
    rmat_error = r_target @ r_current.T
    rotvec_error = rmat2rotvec(rmat_error)* POS_ROT_RATIO
    pos_error_clipped = np.clip(pos_error, -POSITION_CLIPPING, POSITION_CLIPPING)
    rotvec_error_clipped = np.clip(rotvec_error, -ROTATION_CLIPPING, ROTATION_CLIPPING)
    return pos_error_clipped, rotvec_error_clipped