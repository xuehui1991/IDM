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
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

OBJECT_TO_COLORS = {
    # 'unseen'        : 0,
    # 'empty'         : 1,
    #'wall'          : 'grey',
    # 'floor'         : 3,
    # 'door'          : 4,
    #'key'           : 'yellow',
    # 'ball'          : 6,
    # 'box'           : 7,
    'goal'          : 'green',
    # 'lava'          : 9,
    'agent'         : 'red',
}

class VisionIntelligence():

    def __init__(self, env):
        self.env = env

    def detect_object(self, image, first=True):
        obj_pos = collections.OrderedDict()
        key_index = {}

        i = 0
        for key in OBJECT_TO_COLORS:
            pos = self.get_position(image, key)
            # for now dont consider have multiple same objects
            if pos is not None:
                obj_pos[key] = pos
                # assume key will not change
                key_index[key] = i
                i = i+1
        return obj_pos, key_index

    
    def traverse(self, image, thr=2):
        obj_pos, key_index = self.detect_object(image)
        
        agent_pos = obj_pos['agent']
        min_key = None
        min_dis = float('inf')
        min_index = -1
        for key in obj_pos:
            if key is not 'agent':
                dis = self.get_distance(agent_pos, obj_pos[key])
                if min_dis > dis:
                    min_dis = dis
                    min_key = key
        if min_dis < thr:
            min_index = key_index[min_key]

        return min_dis, min_index, key_index[min_key]


    def check_color(self, image, obj_type):
        color = COLORS[OBJECT_TO_COLORS[obj_type]]
        location = np.all((image == color), axis=2)
        return np.where(location == True)


    def get_position(self, image, obj_type):
        """Find the object based on its color."""

        indices = self.check_color(image, obj_type)
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


# def test_env(env_id):
#     env = gym.make(env_id)
#     #env = RGBImgPartialObsWrapper(env)
#     env = RGBImgObsWrapper(env)
#     #env = FullyObsWrapper(env)
#     env = ImgObsWrapper(env)

#     vi = VisionIntelligence(env)

#     #obs = env.reset()
#     obs, reward, done, info = env.step(env.actions.left)
#     # from PIL import Image
#     # im = Image.fromarray(obs)
#     # im.save("obs.png")

#     pos = vi.get_position(obs, "agent")
#     print('agent pos is {}'.format(pos))
#     pos = vi.get_position(obs, "goal")
#     print('goal pos is {}'.format(pos))

#     min_dis, is_reach_obj, min_obj = vi.traverse(obs)
#     print(min_dis, is_reach_obj, min_obj)
    
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--env",
#     help="gym environment to load",
#     default='MiniGrid-Empty-6x6-v0 '
# )
# parser.add_argument(
#     "--seed",
#     type=int,
#     help="random seed to generate the environment with",
#     default=-1
# )
# parser.add_argument(
#     "--tile_size",
#     type=int,
#     help="size at which to render tiles",
#     default=32
# )
# parser.add_argument(
#     '--agent_view',
#     default=False,
#     help="draw the agent sees (partially observable view)",
#     action='store_true'
# )

# args = parser.parse_args()

# test_env(args.env)

# # env = gym.make(args.env)

# # if args.agent_view:
# #     env = RGBImgPartialObsWrapper(env)
# #     env = ImgObsWrapper(env)

# # window = Window('gym_minigrid - ' + args.env)
# # window.reg_key_handler(key_handler)

# # reset()

# # # Blocking event loop
# # window.show(block=True)