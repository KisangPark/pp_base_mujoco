""" MUJOCO VIEWER CLASS """

import mujoco

import sys
import glfw 
import time
from functools import partial

class MUJOCOGLVIEWER():
    def __init__(self, model, data, window_size = (1920, 1080)):

        # initialize with empty lists
        glfw.init()
        self.model = model
        self.data = data

        self.windows = []
        self.contexts = []
        self.scenes = []
        self.cameras = []
        self.options = []
        self.viewport=mujoco.MjrRect(0, 0, 0, 0)
        if not glfw.init():
            sys.exit("couldn't initialize glfw")
        self.last_x, self.last_y = 0, 0
        self.mouse_button = None
        # camera index
        self.fixed_idx = 0
        self.track_idx = 0
        # render interval
        self.render_interval = 0.01
        self.last_render_time = time.time()
        # add default camera
        self.add_cameras(camera_names=["default camera"], types = ['free'], sizes=[window_size])

    def add_cameras(self,camera_names, types, sizes):
        if len(camera_names) != len(sizes) or len(camera_names) != len(types):
            raise ValueError("Length of camera_names, sizes, and types must be the same")
        # adding mujoco camera
        for i, (name, size, type) in enumerate(zip(camera_names, sizes, types)):
            window = glfw.create_window(size[0], size[1], f"Camera: {name}", None, None)
            if not window:
                glfw.terminate()
                raise RuntimeError("GLFW window creation failed")
            glfw.make_context_current(window)
            # scene, camera, options
            scene = mujoco.MjvScene(self.model, maxgeom=1000)
            cam = mujoco.MjvCamera()
            if type == 'free':
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                actual_cursor_pos_callback = partial(self.cursor_pos_callback, scene, cam)
                actual_scroll_callback = partial(self.scroll_callback, scene, cam)
                glfw.set_mouse_button_callback(window, self.mouse_button_callback)
                glfw.set_cursor_pos_callback(window, actual_cursor_pos_callback)
                glfw.set_scroll_callback(window, actual_scroll_callback) 
            elif type == 'fixed':
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = self.fixed_idx
                self.fixed_idx += 1
            elif type == 'track':
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = self.track_idx
                self.track_idx += 1
            opt = mujoco.MjvOption()

            # append to list
            self.scenes.append(scene)
            self.cameras.append(cam)
            self.options.append(opt)
            self.windows.append(window)
            self.contexts.append(mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150))
                 
    
    def is_alive(self):
        if self.model.ncam + 1 < len(self.cameras):
            raise ValueError(f"Number of cameras in model ({self.model.ncam}) is less than the total number of cameras being added ({len(self.cameras)})")
        if any([glfw.window_should_close(win) for win in self.windows]):
            return False
        else:
            return True

    def close(self):
        for window in self.windows:
            glfw.destroy_window(window)
        glfw.terminate()

    def render(self):
        current_time = time.time()
        if current_time - self.last_render_time >= self.render_interval:
            self.last_render_time = current_time

            for i, window in enumerate(self.windows):
            
                glfw.make_context_current(window)
                width, height = glfw.get_framebuffer_size(window)
                self.viewport.width = width
                self.viewport.height = height

                # update scene & render
                mujoco.mjv_updateScene(self.model, self.data, self.options[i], None, self.cameras[i],
                                    mujoco.mjtCatBit.mjCAT_ALL, self.scenes[i])
                # kernel crashes here, mjv update scene
                mujoco.mjr_render(self.viewport, self.scenes[i], self.contexts[i])
                glfw.swap_buffers(window)
            glfw.poll_events()

    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_button = button
        elif action == glfw.RELEASE:
            self.mouse_button = None

    def cursor_pos_callback(self, scene, camera, window, xpos, ypos):
        dx = (xpos - self.last_x)/1000
        dy = (ypos - self.last_y)/1000
        self.last_x, self.last_y = xpos, ypos

        if self.mouse_button is not None:
            action = {
                glfw.MOUSE_BUTTON_LEFT: mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                glfw.MOUSE_BUTTON_RIGHT: mujoco.mjtMouse.mjMOUSE_MOVE_H,
                glfw.MOUSE_BUTTON_MIDDLE: mujoco.mjtMouse.mjMOUSE_ZOOM
            }.get(self.mouse_button, None)
            if action is not None:
                mujoco.mjv_moveCamera(self.model, action, dx, dy, scene, camera)

    def scroll_callback(self, scene, camera, window, xoffset, yoffset):
        # Zoom camera with scroll wheel
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -yoffset/100, scene, camera)
        # print(f"offsets x:{xoffset}, y:{yoffset}")

    """ VIEWER OPTIONS """
    def view_contact_forces(self, show=True):
        for i in range(len(self.options)):
            self.options[i].flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show

    def view_geom(self, group=0, show=True):
        for i in range(len(self.options)):
            self.options[i].geomgroup[group] = show