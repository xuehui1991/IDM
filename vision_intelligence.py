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
    #'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    # 'grey'  : np.array([100, 100, 100]),
    'red'   : np.array([255, 0, 0]),
}

class VisionIntelligence():

    def __init__(self, env):
        self.env = env
        self.COLORS = collections.OrderedDict(COLORS)
        self.obj_pos = collections.OrderedDict()
        self.key_index = collections.OrderedDict()

    def get_goal_size(self, image):
        i = 0
        for color_key in self.COLORS:
            pos = self.get_position(image, color_key)
            # for now dont consider have multiple same objects
            if pos is not None:
                #print(i, color_key)
                self.obj_pos[color_key] = pos
                # assume key will not change
                self.key_index[i] = color_key
                i = i+1

        print('Key index', self.key_index)
        return (len(self.obj_pos)-1)

    def detect_object(self, image):
        i = 0
        for color_key in self.obj_pos:
            pos = self.get_position(image, color_key)
            if pos is not None:
                self.obj_pos[color_key] = pos

    def traverse(self, image, goal_index, thr=11):
        self.detect_object(image)
        agent_pos = self.obj_pos['red']

        # print(goal_index)
        if len(goal_index) == 1:
            goal_index = goal_index[0]
        try:
            goal_color = self.key_index[goal_index]
        except:
            print('Key goal_index error in key index: {}, {}'.format(self.key_index, goal_index))
            raise

        dis = self.get_distance(agent_pos, self.obj_pos[goal_color])

        is_reach = -1
        for index in self.key_index:
            if self.key_index[index] == 'red':
                continue
            color = self.key_index[index]
            d = self.get_distance(agent_pos, self.obj_pos[color])
            if d<thr:
                is_reach = index
        return dis, is_reach

    def check_color(self, image, color_key):
        color = self.COLORS[color_key]
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
