"""
TODO
1) get portable gray map
2) make node object
3) methods to execute RRT star
4) visualize with cv
"""

import cv2
import numpy as np
import random
import heapq
import os
# image value 255: free area
# image value 0:  obstacle area


"""1. define node for rrt star"""
class Node: #class for RRT node
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0.0 #float cost


"""2. drawing function"""
def draw_path(map_image, points):
    #get image, path points -> draw line between each pair!
    for i in range(len(points) - 1):
        cv2.line(map_image, points[i], points[i+1], (0,255,0), 2)
    return map_image



"""
3. functions for RRT
    1) generate random point
    2) get nearest tree vertex
    3) action input -> also named as steer
    4) find Xnear, check cost and update connection (parent)
    5) reverse connection update (child)
    6) collision avoidance?

here, functions about
- generate random
- nearest & Xnear
- steering
"""

def generate_random_point(img, goal, goal_sample_rate):
    #generate random sample from the map
    #get goal according to the sample rate -> biasing

    if random.randint(0, 100) > goal_sample_rate:
        return (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        #tuple of positions -> globally random
    else:
        return goal

def distance(point1, point2): #calculate distance between two points
    #points are 2 dimensional -> 2d norm to get distance
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest_node(nodes, random_point):
    #finding nearest node
    #return node of smallest distance, using lambda key
    return min(nodes, key=lambda node: distance(node.point, random_point))

def Xnear(current, tree_nodes, dist):
    #return tree nodes within the distance
    #current and tree_nodes for node object
    #current means xnew after steering
    Xnear = []
    for node in tree_nodes:
        if distance(current.point, node.point) <= dist:
            Xnear.append(node)
    return Xnear #list of nodes

def steer(tree_node, random_point, control_input):
    #get node, random point, control_input
    #get new node
    #progress forward with fixed value
    if distance(tree_node.point, random_point) < control_input:
        return Node(random_point)

    else: #toward fixed value!
        #get two point, difference to get direction
        #add direction * distance to start point!
        start = np.array(tree_node.point)
        toward = np.array(random_point) - start
        length = np.linalg.norm(toward)
        unit_direction = np.array(toward / length)
        xnew = start + control_input * unit_direction
        #now, xnew is numpy array with point
        #change into integer & tuple
        new_pos = tuple(xnew.astype(int))
        return Node(new_pos) #.astype(int)


"""4. collision check"""
#return true or false

def line_collision_check(first_point, second_point, img):
    #check collision in map
    #check if there is point occupied between two points

    x_values = [first_point[0], second_point[0]]
    y_values = [first_point[1], second_point[1]]
    x_values.sort() #values in increasing order
    y_values.sort()
    for x in range(x_values[0], x_values[1]+1):
        for y in range(y_values[0], y_values[1]+1):
            if img[y, x] == 0:  # occupied value, black is occupied (or 100?)
                return False
    return True

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
        random_point = generate_random_point(img, goal, goal_sample_rate)
        nearest = nearest_node(nodes, random_point)
        new_node = steer(nearest, random_point, control_input=2) #30? change with resolution
        print(new_node)

        #if not collide
        if line_collision_check(nearest.point, new_node.point, img):
            
            #get near set
            near_nodes = Xnear(new_node, nodes, dist=4)#here, Xnear, 60, distance=0.2
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