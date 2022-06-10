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

from agent import *
from util import *
from config import *
from torch.multiprocessing import Pipe


class Runner():

    def __init__(self):
        logger_path = os.path.join('results', f'_{datetime.now().strftime("%m%d%Y_%H%M%S")}')
        self.logger = get_logger(logger_path)
        self.logger.info({section: dict(config[section]) for section in config.sections()})
        print({section: dict(config[section]) for section in config.sections()})
        self.writer = SummaryWriter()

        env_id = default_config['EnvID']
        env_type = default_config['EnvType']
        env = gym.make(env_id)
        env = RGBImgObsWrapper(env)
        self.env = ImgObsWrapper(env)

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
        #print('obs shape:{} action shape:{}'.format(self.input_size, self.output_size))
        self.agent = Agent(self.input_size,
                            self.output_size,
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
        self.step = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.print_interval = 1000
        self.train_interval = 20
        self.loss = []
        self.clean()
        self.train_step = 0



    def clean(self):
        self.total_state = []
        self.total_action = []
        self.total_next_state = []
        self.total_int_reward = []
        self.total_reward = []
        self.total_done = []
        self.total_values = []
        self.total_policy = []

    def post_processing(self, state, action, next_state, reward, done,
                        value, policy, instrice_reward = None):
        self.total_state.append(state)
        self.total_action.append(action)
        self.total_next_state.append(next_state)
        if instrice_reward is not None:
            self.total_int_reward.append(instrice_reward)
        self.total_reward.append(reward)
        self.total_done.append(done)
        self.total_values.append(value)
        self.total_policy.append(policy)


    def get_data(self): 
        total_reward = np.stack(self.total_reward).transpose()
        total_state = np.stack(self.total_state).transpose([0,3,2,1])
        #print('total state shape:{}'.format(total_state.shape))
        total_next_state = np.stack(self.total_next_state).transpose([0,3,2,1])
        total_action = np.stack(self.total_action).transpose().reshape([-1])
        total_done = np.stack(self.total_done).transpose()
        total_values = np.stack(self.total_values).transpose()
        total_policy = self.total_policy
        #print('total policy shape:{}'.format(total_policy.shape))
        #raise Exception('stop here.')
        self.clean()
        return total_state, total_next_state, total_action, total_reward, total_done, total_values, total_policy


    def run(self, seed = 0):
        state = self.env.reset()

        while True:
            action, value, policy = self.agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.step += 1
            self.episode_reward += reward
            # if self.use_icm:
            #     # calculate instric reward
            #     pass

            self.post_processing(state, action, next_state, reward, done, value, policy)
            state = next_state

            # Perform one step of the optimization
            if self.step % self.num_step == 0:
                _, value, _ = self.agent.get_action(state)
                self.total_values.append(value)

                total_state, total_next_state, total_action, total_reward, total_done, total_values, total_policy = self.get_data()

                # Step 3. make target and advantage
                # for now total_int_reward = total_reward
                total_int_reward = total_reward
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
                                            total_action, adv, total_policy)
                self.loss.append(loss)
                self.train_step +=1

                if self.train_step % self.train_interval == 0:
                    #print(self.loss)
                    print('Train step:{} Avg loss:{}'.format(self.train_step, np.mean(self.loss[-10:])))


                #torch.save(agent.model.state_dict(), model_path)

            if done:
                self.episode_step +=1
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0 
                if self.step % self.print_interval:
                    print('Total step:{}  Episode Step:{}  Avg Reward:{}'.format(self.step, self.episode_step, np.mean(self.episode_rewards[-10:])))
                #raise Exception('Stop here.')
                state = self.env.reset()

def main():
    runner = Runner()
    runner.run()


if __name__ == '__main__':
    main()
