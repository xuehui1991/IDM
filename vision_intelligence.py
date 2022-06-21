#!/usr/bin/env python3
import collections
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window


# Map of color names to RGB values
COLORS = {
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'red'   : np.array([255, 0, 0]),
}

class VisionIntelligence():

    def __init__(self, env):
        self.env = env

    def get_goal_size(self, image):
        obj_pos, key_index = self.detect_object(image)
        return (len(obj_pos)-1)

    def detect_object(self, image):
        obj_pos = collections.OrderedDict()
        key_index = {}

        i = 0
        for color_key in COLORS:
            pos = self.get_position(image, color_key)
            # for now dont consider have multiple same objects
            if pos is not None:
                obj_pos[color_key] = pos
                # assume key will not change
                key_index[i] = color_key
                i = i+1
        return obj_pos, key_index

    
    def traverse(self, image, goal_index, thr=2):
        obj_pos, key_index = self.detect_object(image)
        agent_pos = obj_pos['red']

        # print(goal_index)
        if len(goal_index) == 1:
            goal_index = goal_index[0]
        goal_color = key_index[goal_index]
        dis = self.get_distance(agent_pos, obj_pos[goal_color])
        return dis

        # min_key = None
        # min_dis = float('inf')
        # min_index = -1
        # for key in obj_pos:
        #     if key is not 'agent':
        #         dis = self.get_distance(agent_pos, obj_pos[key])
        #         if min_dis > dis:
        #             min_dis = dis
        #             min_key = key
        # if min_dis < thr:
        #     min_index = key_index[min_key]

        # return min_dis, min_index, key_index[min_key]


    def check_color(self, image, color_key):
        color = COLORS[color_key]
        location = np.all((image == color), axis=2)
        return np.where(location == True)


    def get_position(self, image, color_key):
        """
        Find the object based on its color.
        Aussmed one color only contains one object.
        """

        indices = self.check_color(image, color_key)
        #print('indices: ', indices)
        if len(indices[0]) > 0:
            x = (np.amin(indices[1]) + np.amax(indices[1])) // 2
            y = (np.amin(indices[0]) + np.amax(indices[0])) // 2
            # average position
            location = (x, y)
        else:
            #raise Exception('No match object.')
            return None
        return location


    def get_distance(self, l1, l2):
        return np.sqrt(np.sum(np.square(np.asarray(l1) - np.asarray(l2))))
