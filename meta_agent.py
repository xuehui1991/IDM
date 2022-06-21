import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import MetaControllerNetwork, ControllerNetwork

class MetaAgent():

    def __init__(self,
                input_size,
                output_size,
                goal_size,
                num_step,
                gamma,
                lam=0.95,
                learning_rate=1e-4,
                ent_coef=0.01,
                clip_grad_norm=0.5,
                epoch=3,
                batch_size=128,
                ppo_eps=0.1,
                eta=0.01,
                use_gae=True,
                use_cuda=False,
                use_noisy_net=False):

        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma

        self.goal_size = goal_size

        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        #self.use_gae = use_gae
        #self.ent_coef = ent_coef
        #self.eta = eta
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # TODO: change the learning rate
        self.meta_model = MetaControllerNetwork(input_size, goal_size)
        self.meta_optimizer = optim.Adam(list(self.meta_model.parameters()), lr=learning_rate)
        self.meta_model = self.meta_model.to(self.device)

        self.model = ControllerNetwork(input_size, output_size, goal_size)
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=learning_rate)
        self.model = self.model.to(self.device)


    def get_goal(self, state, eval=False):
        state = torch.Tensor(state).to(self.device).float()
        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0).transpose(1, 3)
            
        policy = self.meta_model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
        if not eval:
            action = self.random_choice_prob_index(action_prob)
        else:
            # need to consider here, shall we try the goal
            action = np.argmax(action_prob, axis=1)

        return action, policy.detach()        


    def get_action(self, state, goal, eval=False):
        state = torch.Tensor(state).to(self.device).float()
        goal = torch.Tensor(goal).to(self.device).float()
        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0).transpose(1, 3)
        
        assert len(goal.shape) == 2
            
        policy, value = self.model(state, goal)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
        if not eval:
            action = self.random_choice_prob_index(action_prob)
        else:
            action = np.argmax(action_prob, axis=1)

        return action, value.data.cpu().numpy().squeeze(), policy.detach()


    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    
    def compute_intrinsic_reward(self, state, next_state, action):
        pass

    def train_meta_model(self, s_batch, action_batch, reward_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        action_batch = torch.squeeze(torch.LongTensor(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        sample_range = np.arange(len(action_batch))
        # print('action_batch shape:', action_batch.shape)
        # print('s_batch shape:', s_batch.shape)
        # print('reward_batch shape:', reward_batch.shape)

        meta_loss = []
        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(action_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                policy = self.meta_model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(action_batch[sample_idx])      
                # print('log_prob shape:', log_prob.shape)          

                self.meta_optimizer.zero_grad()
                loss = - (log_prob* reward_batch[sample_idx]).mean()
                meta_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.meta_optimizer.step()
            
        return np.mean(meta_loss)

    def train_model(self, s_batch, next_s_batch, target_batch, y_batch, adv_batch, old_policy, goal_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        goal_batch = torch.FloatTensor(goal_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        r_loss = []
        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                # action_onehot = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)
                # action_onehot.zero_()
                # action_onehot.scatter_(1, y_batch[sample_idx].view(-1, 1), 1)
                # real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                #     [s_batch[sample_idx], next_s_batch[sample_idx], action_onehot])

                # inverse_loss = ce(
                #     pred_action, y_batch[sample_idx])

                # forward_loss = forward_mse(
                #     pred_next_state_feature, real_next_state_feature.detach())
                # ---------------------------------------------------------------------------------

                policy, value = self.model(s_batch[sample_idx], goal_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = (actor_loss + 0.5 * critic_loss - 0.001 * entropy) 
                r_loss.append(loss.item())
                #+ forward_loss + inverse_loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        return np.mean(r_loss)