import numpy as np
import copy
import os
import logging
import sys
import gym
import gym_minigrid
from gym_minigrid.wrappers import *

from tensorboardX import SummaryWriter
from datetime import datetime

from util import get_logger

#from agent import *
from meta_agent import *
from util import *
from config import *
# from test_minigrid import *
from vision_intelligence import *
from distance_predictor import *
from torch.multiprocessing import Pipe

import gym.wrappers


class Runner():

    def __init__(self):
        # logger_path = os.path.join('results', f'_{datetime.now().strftime("%m%d%Y_%H%M%S")}')
        # self.logger = get_logger(logger_path)
        # self.logger.info({section: dict(config[section]) for section in config.sections()})
        print({section: dict(config[section]) for section in config.sections()})
        self.writer = SummaryWriter()

        env_id = default_config['EnvID']
        env_type = default_config['EnvType']
        env = gym.make(env_id)
        env = RGBImgObsWrapper(env)
        self.env = ImgObsWrapper(env)
        #self.env = gym.wrappers.Monitor(env=self.env, directory="./videos", force=True)

        self.use_cuda = default_config.getboolean('UseGPU')
        self.use_gae = default_config.getboolean('UseGAE')
        self.use_noisy_net = default_config.getboolean('UseNoisyNet')
        self.lam = float(default_config['Lambda'])
        self.num_step = int(default_config['NumStep'])
        self.ppo_eps = float(default_config['PPOEps'])
        self.epoch = int(default_config['Epoch'])
        self.mini_batch = int(default_config['MiniBatch'])
        self.batch_size = int(self.num_step * 1 / self.mini_batch)
        self.learning_rate = float(default_config['LearningRate'])
        self.entropy_coef = float(default_config['Entropy'])
        self.gamma = float(default_config['Gamma'])
        self.eta = float(default_config['ETA'])
        self.clip_grad_norm = float(default_config['ClipGradNorm'])

        self.input_size = env.observation_space.shape  # 4
        self.output_size = env.action_space.n  # 2

        self.vi = VisionIntelligence(self.env)
        state = self.env.reset()
        self.goal_size = self.vi.get_goal_size(state)
        print('Init env {} and the goal size is {}'.format(env_id, self.goal_size))

        # TODO: change for envoriment
        self.num_meta_eps = 50

        #print('obs shape:{} action shape:{}'.format(self.input_size, self.output_size))
        self.agent = MetaAgent(self.input_size,
                            self.output_size,
                            self.goal_size,
                            self.num_step,
                            self.gamma,
                            lam=self.lam,
                            learning_rate=self.learning_rate,
                            ent_coef=self.entropy_coef,
                            clip_grad_norm=self.clip_grad_norm,
                            epoch=self.epoch,
                            batch_size=self.batch_size,
                            ppo_eps=self.ppo_eps,
                            eta=self.eta,
                            use_cuda=self.use_cuda,
                            use_gae=self.use_gae,
                            use_noisy_net=self.use_noisy_net)

        print('Init agent done.')

        self.step = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.print_interval = 100
        self.train_interval = 20
        self.save_interval = 20
        self.loss = []
        
        self.clean()
        self.clean_for_predictor()
        self.clean_for_meta()

        self.train_step = 0

        #self.predictor = DistancePredictor(self.input_size, 1, self.use_cuda, self.learning_rate)
        self.p_loss = []
        self.m_loss =[]

        self.eps_state = []
        self.eps_goal = []
        self.eps_reward = []

        print('Init done.')

    def clean_for_meta(self):
        self.eps_states = []
        self.eps_goals = []
        self.eps_rewards = []


    def clean(self):
        self.total_state = []
        self.total_action = []
        self.total_next_state = []
        self.total_int_reward = []
        self.total_reward = []
        self.total_done = []
        self.total_values = []
        self.total_policy = []

        self.total_goal = []


    def clean_for_predictor(self):
        self.cur_state = []

    def generate_predictor_data(self):
        X = []
        Y = []
        i = 0
        for state in self.cur_state[::-1]:
            X.append(state)
            Y.append(i)
            i= i + 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        p = np.random.permutation(range(len(X)))
        X,Y = X[p],Y[p]
        Y = np.expand_dims(Y, 1)
        return X, Y

    def train_predictor(self):
        X, Y = self.generate_predictor_data()
        X = np.stack(X).transpose([0,3,2,1])
        #print('X shape:{} Y shape:{}'.format(X.shape, Y.shape))
        self.cur_state = []
        loss = self.predictor.fit(X, Y)
        return loss


    def post_processing_for_meta(self, state, goal, reward):
        self.eps_goal.append(goal)
        self.eps_state.append(state)
        self.eps_reward.append(reward)


    def refine_data_for_meta(self):
        self.eps_states.append(np.asarray(self.eps_state))
        self.eps_goals.append(np.asarray(self.eps_goal))
        reward = []
        R = 0
        for r in self.eps_reward[::-1]:
            R = r + self.gamma * R
            reward.insert(0,R)
        reward = (reward - np.mean(reward)) / (np.std(reward) + 0.0001)
        self.eps_rewards.append(np.asarray(reward))

        self.eps_goal = []
        self.eps_state = []
        self.eps_reward = []


    def post_processing(self, state, action, next_state, reward, done,
                        value, policy, instrice_reward = None, goal=None):
        self.total_state.append(state)
        self.total_action.append(action)
        self.total_next_state.append(next_state)
        if instrice_reward is not None:
            self.total_int_reward.append(instrice_reward)
        self.total_reward.append(reward)
        self.total_done.append(done)
        self.total_values.append(value)
        self.total_policy.append(policy)

        self.cur_state.append(state)

        #print('post prcessing:', goal)
        if goal is not None:
            self.total_goal.append(np.squeeze(goal, axis = 0))


    def get_data_for_meta(self):
        total_eps_state =  np.concatenate(self.eps_states, axis=0).transpose([0,3,2,1])
        total_eps_goal =  np.concatenate(self.eps_goals, axis=0)
        total_eps_reward = np.concatenate(self.eps_rewards, axis=0)

        self.clean_for_meta()
        return total_eps_state, total_eps_goal, total_eps_reward 


    def get_data(self): 
        total_int_reward = np.stack(self.total_int_reward).transpose()
        total_reward = np.stack(self.total_reward).transpose()
        total_state = np.stack(self.total_state).transpose([0,3,2,1])
        total_next_state = np.stack(self.total_next_state).transpose([0,3,2,1])
        total_action = np.stack(self.total_action).transpose().reshape([-1])
        total_done = np.stack(self.total_done).transpose()
        total_values = np.stack(self.total_values).transpose()
        total_policy = self.total_policy
        #print('total_goal shape #1 is {} each shape is {}'.format(len(self.total_goal), self.total_goal[0].shape))
        total_goal = np.stack(self.total_goal)
        #print('total_goal shape #2 is {}'.format(total_goal.shape))
        self.clean()
        return total_state, total_next_state, total_action, total_reward, total_done, total_values, total_policy, total_int_reward, total_goal


    def train_controller(self, state, goal):
        _, value, _ = self.agent.get_action(state, goal)
        self.total_values.append(value)

        total_state, total_next_state, total_action, total_reward, total_done,\
            total_values, total_policy, total_int_reward, total_goal = self.get_data()
        #print('shape of total goal is:{}'.format(total_goal.shape))

        #total_int_reward = total_reward + total_int_reward
        total_int_reward = total_int_reward
        target, adv = make_train_data_v1(total_int_reward,
                                    np.zeros_like(total_int_reward),
                                    total_values,
                                    self.gamma,
                                    self.num_step,
                                    num_worker=1,
                                    use_gae=self.use_gae,
                                    lam=self.lam)

        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        # -----------------------------------------------
        # Step 5. Training!
        loss  = self.agent.train_model(total_state, total_next_state, target, 
                                    total_action, adv, total_policy, total_goal)

        return loss

    
    def train_meta_controller(self):
        total_eps_state, total_eps_goal, total_eps_reward  = self.get_data_for_meta()
        meta_loss = self.agent.train_meta_model(total_eps_state, total_eps_goal, total_eps_reward)
        return meta_loss


    def get_int_reward(self, goal, state=None, next_state=None, ext_reward=None):
        if goal == 0.0:
            assert ext_reward is not None
            return ext_reward
        else:
            assert state is not None
            assert next_state is not None
            r1, _, _ = self.vi.traverse(state, goal)
            r2, _, _ = self.vi.traverse(next_state, goal)
            int_reward = (r1 - r2)/100
            return int_reward


    def run(self, seed = 0):
        #env.seed(seed)
        state = self.env.reset()

        goal, _ = self.agent.get_goal(state)

        while True:
            #action, value, policy = self.agent.get_action(state)
            if len(goal.shape) !=2:
                goal = goal[np.newaxis, :]
            action, value, policy = self.agent.get_action(state, goal)

            next_state, reward, done, _ = self.env.step(action)

            self.step += 1
            self.episode_reward += reward
            # if self.use_icm:
            #     # calculate instric reward

            # r1, _, _ = self.vi.traverse(state)
            # r2, _, _ = self.vi.traverse(next_state)
            # int_reward = (r1 - r2)/100
            # TODO:  will change to the distance predictor later...
            int_reward = self.get_int_reward(goal, state, next_state, reward)

            self.post_processing_for_meta(state, goal, reward)
            self.post_processing(state, action, next_state, reward, done, value, policy, instrice_reward = int_reward, goal=goal)
            state = next_state

            # Perform one step of the optimization
            if self.step % self.num_step == 0:
                print('Total step:{}, Eps reward: {}'.format(self.step, self.episode_reward))

                loss = self.train_controller(state, goal)
                self.loss.append(loss)

                if self.train_step % self.train_interval == 0:
                    print('Train step:{} Avg loss:{}'.format(self.train_step, np.mean(self.loss[-10:])))
            
                self.train_step +=1

            if done:
                self.episode_step +=1
                self.episode_rewards.append(self.episode_reward)
                if self.step % self.print_interval:
                    # print('Total Step:{}  Episode Step:{}  Avg Reward:{} Predictor Loss:{} '.format(self.step, 
                    #             self.episode_step, np.mean(self.episode_rewards[-10:]), np.mean(self.p_loss[-10:])))
                    print('Total Step:{}  Episode Step:{}  Avg Reward:{} '.format(self.step, 
                                self.episode_step, np.mean(self.episode_rewards[-10:])))

                # loss = self.train_predictor()
                # self.p_loss.append(loss)
                
                self.refine_data_for_meta()
                if len(self.episode_rewards) % self.num_meta_eps == 0:
                    self.m_loss.append(self.train_meta_controller())
                    print('Total EP:{}  Episode Step:{}  Meta Loss:{} '.format(self.step, 
                                self.episode_step, np.mean(self.m_loss[-1:])))

                state = self.env.reset()
                self.episode_reward = 0 


def main():
    runner = Runner()
    runner.run()
    #runner.eval()


if __name__ == '__main__':
    main()
