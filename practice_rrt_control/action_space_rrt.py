""" JOINT SPACE RRT """

"""
Action-space RRT
1) Overview of algorithm
    - 6-dimension action space, 6d path point
    - simple implementation: initial & goal pose given
    - Problem: given end-effector 6d pose
        - multiple points possible (infinite)
        - get volumetric "ending zone" with rough FK
"""


from ast import AsyncFunctionDef
import cv2
import numpy as np
import random
import heapq
import os


""" Utility functions """
def distance(point1, point2): #calculate distance between two points
    #points are 2 dimensional -> 2d norm to get distance
    return np.linalg.norm(np.array(point1) - np.array(point2))



""" Simple RRT Node """
class Node: #class for RRT node
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0.0 #float cost



""" Joint Space RRT Class """
class JointSpace_RRT():

    def __init__(self, mujoco_model, mujoco_data):

        self.model = mujoco_model
        self.data = mujoco_data
        self.dim = self.model.nu # num of actuators

        self.nodes = []
        

    def update_mj_model_data(self, mujoco_model, mujoco_data):
        self.model = mujoco_model
        self.data = mujoco_data

    def init_rrt_params(self, xnear_dist, steer_dist, rewire_dist, unit_step):
        self.xnear_dist = xnear_dist
        self.steer_dist = steer_dist
        self.rewire_dist = rewire_dist

        # get control range
        self.ctrl_min = self.model.actuator_ctrlrange[:,0]
        self.ctrl_max = self.model.actuator_ctrlrange[:,1]

        self.unit_step = unit_step
        self.resolution = [int((max-min)/unit_step) for max, min in zip(self.ctrl_min, self.ctrl_max)] # joint space resolution, number of steps -> should be an array of self.dim
        print(f"resolution: {self.resolution}")

    def generate_6d_space(self):
        """
        Generate a 6-dimensional grid space.

        Args:
            min_vals (list or array): 6 minimum values, one for each dimension.
            max_vals (list or array): 6 maximum values, one for each dimension.
            resolution (int): number of points per dimension (same for all).

        Returns:
            np.ndarray: array of shape (resolution**6, 6),
                        each row is one 6D point [x1, x2, x3, x4, x5, x6].
        """
        # get min & max from mujoco data
        min_vals = self.model.actuator_ctrlrange[:,0]
        max_vals = self.model.actuator_ctrlrange[:,1]

        # Create 1D grids for each dimension
        axes = [np.linspace(min_vals[i], max_vals[i], self.resolution) for i in range(6)]
        # Create full 6D grid (each array has shape [res, res, res, res, res, res])
        mesh = np.meshgrid(*axes, indexing="ij")
        # Stack and reshape into (N, 6)
        grid_points = np.stack(mesh, axis=-1).reshape(-1, 6)

        return grid_points

    def map_pose_to_space(self, pose = np.zeros(6)):
        # map robot pose into 6d space
        pass
    

    def node_collision_check(self, node_1, node_2):
        # 6d collision check
        #check collision in map
        #check if there is point occupied between two points

        x_values = [node.point[0] for node in]
        y_values = [first_point[1], second_point[1]]
        x_values.sort() #values in increasing order
        y_values.sort()
        for x in range(x_values[0], x_values[1]+1):
            for y in range(y_values[0], y_values[1]+1):
                if img[y, x] == 0:  # occupied value, black is occupied (or 100?)
                    return False
        return True






def bresenham_6d_collision(space, point_a, point_b):
    """
    Check collision along the 6D line connecting two integer grid points
    using a generalized Bresenham algorithm.

    Args:
        space (np.ndarray): 6D occupancy grid (values 0 or 1)
        point_a (array-like): start 6D point in grid index coordinates (ints)
        point_b (array-like): end 6D point in grid index coordinates (ints)

    Returns:
        bool: True if collision (hits any occupied voxel), else False
    """
    p0 = np.array(point_a, dtype=int)
    p1 = np.array(point_b, dtype=int)
    dims = self.dim

    # Delta and step
    delta = np.abs(p1 - p0)
    step = np.sign(p1 - p0)

    # Major axis = dimension with largest delta (index)
    major_axis = np.argmax(delta)

    # Initialize error terms
    err = np.zeros(dims, dtype=float)
    for i in range(dims):
        if i != major_axis:
            err[i] = delta[major_axis] / 2.0

    # Current position
    pos = p0.copy()

    # Number of steps along the major axis
    n_steps = delta[major_axis] + 1

    for _ in range(n_steps):
        # Check collision
        if space[tuple(pos)] == 1:
            return True

        for i in range(dims):
            if i == major_axis:
                continue
            err[i] -= delta[i]
            if err[i] < 0:
                pos[i] += step[i]
                err[i] += delta[major_axis]
        pos[major_axis] += step[major_axis]

    return False
















    def get_random_point(self, goal):

        if random.randint(0, 100) > self.goal_sample_rate: # sample random point
            # return 6d random point
            point = np.zeros(self.dim)
            for i, value in enumerate(self.space.shape):
                point[i] = value
            return point
            #tuple of positions -> globally random
        else:
            return goal

    def get_nearest_node(self, point): # ** returns Node **
        #return node of smallest distance, using lambda key
        node_curr = min(self.nodes, key=lambda node: distance(node.point, point))
        return node_curr

    def steer(self, nearest_node, point):
        # check steering for random point, returns node if feasible
        if distance(nearest_node.point, point) < self.steer.dist:
            return Node(point)
        else:
            # far than distance, move by control input
            start = np.array(tree_node.point)
            toward = np.array(random_point) - start
            length = np.linalg.norm(toward)
            unit_direction = np.array(toward / length)
            xnew = start + control_input * unit_direction
            #now, xnew is numpy array with point
            #change into integer & tuple
            new_pos = tuple(xnew.astype(int))
            return Node(new_pos) #.astype(int)


    def get_xnear_nodes(self, node_curr):
        #return tree nodes within the distance
        Xnear = []
        for node in self.nodes:
            if distance(node_curr.point, node.point) <= self.xnear_dist:
                Xnear.append(node)
        return Xnear #list of nodes  



"""
RRT functions
"""







"""4. collision check"""
#return true or false



    #currently, detecting all objects within the square


"""
5. connection update
    1) pick parent
    2) update near nodes (forward & backward)
"""
#pick parent & forward
#after choosing xnew with steering, compare costs within Xnear

def choose_parent(current, Xnear, image):
    #choose lowest cost within Xnear

    #if no nodes in Xnear, return none (no parent)
    if not Xnear:
        return None

    #if nodes exist, compare costs
    min_cost = float('inf')
    for node in Xnear: #near nodes? -> by Xnear function
        if line_collision_check(node.point, current.point, image): #true if no collision
            cost = node.cost + distance(node.point, current.point) #cost for distance
            #if using other costs, I can modify here to make changes!

            if cost < min_cost:
                min_cost = cost
                current.parent = node
                current.cost = cost
    return current #if new_node.parent else None


#backward rewiring
#parent already chosen from upper function
#-> need current node and Xnear, compare Xnear costs and rewire!

def rewire(current, Xnear, image):

    for node in Xnear:
        if node != current.parent: #no need to deal with parent
            if line_collision_check(node.point, current.point, image) and current.cost + distance(current.point, node.point) < node.cost:
                node.parent = current
                node.cost = current.cost + distance(current.point, node.point)



"""
6. RRT search -> overall !
    1) get image, start, goal, iteration, goal sample rate (percent)
    2) get random point
    3) collision check -> steer
    4) Xnear
    5) choose parent with X near
    6) rewire
"""

def RRT_STAR_search(img, start, goal, iter_max=5000, goal_sample_rate=20):

    #1. define node list
    nodes = [Node(start)]
    
    for i in range(iter_max):
        #get random, nearest, xnew
        random_point = get_random_point(img, goal, goal_sample_rate)
        nearest = get_nearest_node(nodes, random_point)
        new_node = steer(nearest, random_point, control_input=2) #30? change with resolution
        print(new_node)

        #if not collide
        if line_collision_check(nearest.point, new_node.point, img):
            
            #get near set
            near_nodes = get_xnear_nodes(new_node, nodes, dist=4)#here, Xnear, 60, distance=0.2
            #update parent
            new_node = choose_parent(new_node, near_nodes, img)
            #if parent connected
            if new_node:
                nodes.append(new_node)
                rewire(new_node, near_nodes, img)

        #if collide -> no action

        #terminalize
        if distance(new_node.point, goal) <= 2:  # 목표에 가까워진 경우 종료
            return get_path(new_node)



def get_path(node):
    #extract path
    path = []
    while node:
        path.append(node.point)
        node = node.parent
    return path[::-1]  # reverse order


## for test
def draw_point(map_image):
    #draw point
    cv2.line(map_image, (30,0), (125,80), (0,255,0), 10)
    return map_image


"""main function"""


if __name__ == '__main__':
    # 이미지 로드 (0: 이동 불가능, 255: 이동 가능)
    #resolution: 20cm

    #load image: resolution 0.05
    img = cv2.imread('/home/kisangpark/map.pgm', cv2.IMREAD_UNCHANGED)
    print('width=',img.shape[1],' height=',img.shape[0])
    # 시작점과 목적지 설정
    # 1000,1000  => 1270, 600
    start_point = (25,68)  # 시작점 좌표
    goal_point = (90,25)  # 목적지 좌표

    path = RRT_STAR_search(img, start_point, goal_point)
    path_modified = np.array(path) * 0.05 #resolution?
    # 결과 출력
    print("final Path:", path)
    img= draw_path(img, path)
    #img = draw_point(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()