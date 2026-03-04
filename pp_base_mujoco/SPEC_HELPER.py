""" IMPORTS """

import os
import sys
import numpy as np
import time
import xml.etree.ElementTree as ET
from lxml import etree

import mujoco
from pp_base_mujoco.UTILS import *

class MJSPECHELPER:
    def __init__(
            self,
            xml_path = "../asset/floor_white_gray.xml"
            ): 
        self.spec = mujoco.MjSpec.from_file(xml_path)

    def add_robot(
            self,
            path,
            body_name = "base_link",
            p=(0, 0, 0),
            r=(0, 0, 0),
            prefix = "",
            suffix = ""
    ):
        """ 
        Add robot to MjSpec
        Args:
            - path: path to robot xml file
            - p: position of the robot
            - r: rotation of the robot
            - prefix: prefix for the robot name
            - suffix: suffix for the robot name
        """

        frame = self.spec.worldbody.add_frame(pos=p, euler = r)
        robot_spec = mujoco.MjSpec.from_file(path)
        frame.attach_body(
            robot_spec.body(body_name), # assume that each xml has worldbody element 
            prefix=prefix,
            suffix=suffix
            )
    
    def add_geom(
            self,
            type,
            size,
            name = "",
            freejoint=False,
            p=(0, 0, 0),
            r=(0, 0, 0),
            rgba=(0.3, 0.3, 0.3, 0.5),
            group=0,
            friction=(1.0, 0.005, 0.0001),
            mass=1.0 # 1kg
    ):
        """ 
        Add geometry to MjSpec, with body encapsulation
        Args:
            - type: type of the geometry (e.g., "box", "sphere", "cylinder")
            - size: size of the geometry (e.g., for box: [length, width, height], for sphere: [radius], for cylinder: [radius, height])
            - freejoint: whether the geometry is freejoint
            - p: position of the geometry
            - r: rotation of the geometry
            - name: name of the geometry
            - rgba: color and transparency of the geometry (optional)
        """
        body = self.spec.worldbody.add_body(
            name=name,
            pos=p,
            euler=r
        )
        if freejoint:
            body.add_freejoint()
        if type == "box":
            body.add_geom(
                type       = mujoco.mjtGeom.mjGEOM_BOX,
                size       = size,
                pos        = np.zeros((3)),
                euler      = [0, 0, 0],
                rgba       = rgba
            )
        elif type == "sphere":
            body.add_geom(
                type       = mujoco.mjtGeom.mjGEOM_SPHERE,
                size       = size,
                pos        = np.zeros((3)),
                euler      = [0, 0, 0],
                rgba       = rgba
            )
        elif type == "cylinder":
            body.add_geom(
                type       = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size       = size,
                pos        = np.zeros((3)),
                euler      = [0, 0, 0],
                rgba       = rgba
            )
        else:
            raise ValueError("Unsupported geometry type: {}".format(type))
        
    def compile(self):
        """ Compile the MjSpec to check for errors and prepare for simulation """
        model = self.spec.compile()
        data = mujoco.MjData(model)
        return model, data
    
    def print_xml(self):
        """ Print the XML representation of the MjSpec """
        xml_str = self.spec.to_xml()
        if isinstance(xml_str, ET.Element):
            rough_string = ET.tostring(xml_str, encoding='unicode')
        else:
            rough_string = xml_str

        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.fromstring(rough_string, parser=parser)
        pretty_xml = etree.tostring(tree, pretty_print=True, encoding='unicode')
        print(pretty_xml)