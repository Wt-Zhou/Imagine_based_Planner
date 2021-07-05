# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agent.World_model.transition_model import make_transition_model


class World_Model(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        state_space_dim,
        transition_model_type,
        env,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        decoder_lr=0.0005,
        decoder_weight_lambda=0.0,
        bisim_coef=0.5
    ):
        self.device = device
        self.discount = discount
        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef
        self.state_space_dim = state_space_dim

        self.action_shape = action_shape
        self.transition_model = make_transition_model(
            transition_model_type, state_space_dim, action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
        nn.Linear(state_space_dim + action_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)).to(device)

        # optimizer for decoder
        self.reward_decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )
        self.train()
        self.env = env

    def train(self, training=True):
        self.training = training
    
    def update_transition_reward_model(self, obs, action, next_obs, reward,  step):
        obs_with_action = torch.cat([obs, action], dim=1)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(obs_with_action)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        diff = (pred_next_latent_mu - next_obs.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        # L.log('train_ae/transition_loss', loss, step)

        pred_next_reward = self.reward_decoder(obs_with_action)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        # print("pred_next_reward",pred_next_reward)
        # print("reward",reward)
        # print("reward_loss",reward_loss)
        total_loss = loss + reward_loss
        return total_loss,loss,reward_loss

    def update(self, replay_buffer, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        # L.log('train/batch_reward', reward.mean(), step)

        transition_reward_loss,loss,reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, step)
        total_loss = transition_reward_loss
        self.reward_decoder_optimizer.zero_grad()
        total_loss.backward()
        self.reward_decoder_optimizer.step()

        with open("Reward_loss.txt", 'a') as fw:
            fw.write(str(loss.detach().cpu().numpy())) 
            fw.write(", ")
            fw.write(str(reward_loss.detach().cpu().numpy())) 
            fw.write("\n")
            fw.close()    

        print("[World_Model] : Updated all models! Step:",step)

    def get_reward_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)

        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1)
            return self.reward_decoder(obs_with_action)

    def get_trans_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)
        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1)
            return self.transition_model(obs_with_action)

    def calculate_bisimulation_pess(self, state_corner, state_normal, action_normal):
        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], state_corner)
        state_corner = torch.as_tensor(np_obs, device=self.device).float()
        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], state_normal)
        state_normal = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action_normal)
        action_normal = torch.as_tensor(np_action, device=self.device)
        with torch.no_grad():
            bisim_for_corner_action = []
            for action in self.action_shape:
                np_action = np.empty((1, 1), dtype=np.float32)
                np.copyto(np_action[0], action)
                action = torch.as_tensor(np_action, device=self.device)

                obs_with_action = torch.cat([state_normal, action_normal], dim=1)
                normal_reward = self.reward_decoder(obs_with_action)

                obs_with_action = torch.cat([state_corner, action], dim=1)
                corner_reward = self.reward_decoder(obs_with_action)
                r_dist = F.smooth_l1_loss(normal_reward, corner_reward, reduction='none')

                pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([state_normal, action_normal], dim=1))
                pred_next_latent_mu2, pred_next_latent_sigma2 = self.transition_model(torch.cat([state_corner, action], dim=1))

                transition_dist = torch.sqrt((pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) + (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2))
                bisim_for_corner_action.append(r_dist + self.discount * transition_dist)

        
        max_action = bisim_for_corner_action.index(max(bisim_for_corner_action))
        return bisim_for_corner_action[max_action], max_action, r_dist, transition_dist

    def calculate_bisimulation_optimal(self, state_corner, state_normal, action_normal):
        with torch.no_grad():
            bisim_for_corner_action = []
            for action in self.action_shape:
                obs_with_action = state_normal.append(action_normal)
                normal_reward = self.reward_decoder(obs_with_action)

                obs_with_action = state_corner.append(action)
                corner_reward = self.reward_decoder(obs_with_action)
                r_dist = F.smooth_l1_loss(normal_reward, corner_reward, reduction='none')

                pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([state_normal, action_normal], dim=1))
                pred_next_latent_mu2, pred_next_latent_sigma2 = self.transition_model(torch.cat([state_corner, action], dim=1))

                transition_dist = torch.sqrt((pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) + (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2))
                bisim_for_corner_action.append(r_dist + self.discount * transition_dist)

        
        min_action = bisim_for_corner_action.index(min(bisim_for_corner_action))
        return bisim_for_corner_action[min_action], min_action

            
    def save(self, model_dir, step):
        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.transition_model.state_dict(),
            '%s/transition_model%s.pt' % (model_dir, step)
        )


    def load(self, model_dir, step):

        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        )
        self.transition_model.load_state_dict(
            torch.load('%s/transition_model%s.pt' % (model_dir, step))
        )


