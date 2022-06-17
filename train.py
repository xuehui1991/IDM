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
from test_minigrid import *
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
        self.env = gym.wrappers.Monitor(env=self.env, directory="./videos", force=True)

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

        self.model_dir = default_config['ModelDir']
        self.model_name = default_config['ModelName']
        self.load_model = default_config['LoadModel']


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
        if self.load_model != 'NA':
            print(self.load_model)
            model_path = os.path.join(self.model_dir, self.model_name + self.load_model + '.pt')
            self.agent.model.load_state_dict(torch.load(model_path))

        self.step = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.print_interval = 1000
        self.train_interval = 20
        self.save_interval = 20
        self.loss = []
        self.clean()
        self.clean_for_predictor()
        self.train_step = 0

        self.vi = VisionIntelligence(self.env)
        self.predictor = DistancePredictor(self.input_size, 1, self.use_cuda, self.learning_rate)
        self.p_loss = []


    def clean(self):
        self.total_state = []
        self.total_action = []
        self.total_next_state = []
        self.total_int_reward = []
        self.total_reward = []
        self.total_done = []
        self.total_values = []
        self.total_policy = []


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

        self.cur_state.append(state)


    def get_data(self): 
        total_int_reward = np.stack(self.total_int_reward).transpose()
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
        return total_state, total_next_state, total_action, total_reward, total_done, total_values, total_policy, total_int_reward

    # def load_model(self, model, model_path):
    #     model.load_state_dict(torch.load(model_path))
    
    def save_model(self, model, step, model_name, model_dir):
        model_path = os.path.join(model_dir, model_name + str(step) + '.pt')
        torch.save(model.state_dict(), model_path)


    def eval(self, seed = 0, eval_eps=10):
        #env.seed(seed)
        state = self.env.reset()
        episode_step = 0
        episode_reward = 0
        episode_rewards = []
        
        while True:
            action, value, policy = self.agent.get_action(state, eval=True)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            # r1, _, _ = self.vi.traverse(state)
            # r2, _, _ = self.vi.traverse(next_state)
            # int_reward = (r1 - r2)/100
            # self.post_processing(state, action, next_state, reward, done, value, policy, instrice_reward = int_reward)
            
            state = next_state

            if done:
                episode_step +=1
                episode_rewards.append(episode_reward)
                episode_reward = 0 

                if episode_step > eval_eps:
                    print('Episode Step:{}  Avg Reward:{}'.format(episode_step, np.mean(episode_rewards)))
                    break
                state = self.env.reset()


    def run(self, seed = 0):
        #env.seed(seed)
        state = self.env.reset()

        while True:
            action, value, policy = self.agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.step += 1
            self.episode_reward += reward
            # if self.use_icm:
            #     # calculate instric reward
            #     pass

            r1, _, _ = self.vi.traverse(state)
            r2, _, _ = self.vi.traverse(next_state)
            int_reward = (r1 - r2)/100
            self.post_processing(state, action, next_state, reward, done, value, policy, instrice_reward = int_reward)
            state = next_state

            # Perform one step of the optimization
            if self.step % self.num_step == 0:
                _, value, _ = self.agent.get_action(state)
                self.total_values.append(value)

                total_state, total_next_state, total_action, total_reward, total_done, total_values, total_policy, total_int_reward = self.get_data()

                # Step 3. make target and advantage
                # for now total_int_reward = total_reward
                total_int_reward = total_reward + total_int_reward
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

                if self.train_step % self.train_interval == 0:
                    #print(self.loss)
                    print('Train step:{} Avg loss:{}'.format(self.train_step, np.mean(self.loss[-10:])))
                    if self.train_step % self.save_interval ==0:
                        self.save_model(self.agent.model, self.train_step, self.model_name, self.model_dir)
                        #raise

                self.train_step +=1

            if done:
                self.episode_step +=1
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0 
                if self.step % self.print_interval:
                    #print('true reward: {}\t int reward: {}'.format(reward, int_reward))
                    print('Total Step:{}  Episode Step:{}  Avg Reward:{} Predictor Loss:{} '.format(self.step, 
                                self.episode_step, np.mean(self.episode_rewards[-10:]), np.mean(self.p_loss[-10:])))
                
                loss = self.train_predictor()
                self.p_loss.append(loss)
                state = self.env.reset()

def main():
    runner = Runner()
    runner.run()
    #runner.eval()


if __name__ == '__main__':
    main()
