import gym
import numpy as np
import gym_routing
import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_03_Cut_In
from Agent.Imagine_Planner import GMBRL
from Agent.world_model.self_attention.self_atten_world_model import GNN_World_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = CarEnv_03_Cut_In()
world_model = GNN_World_Model(obs_shape=env.observation_space.shape,
            action_shape=[1], # discrete, 1 dimension!
            state_space_dim=env.state_dimension,
            device=device,
            env=env)
world_model.train_world_model(env)
# model = GMBRL(world_model)
# model.update_world_model(env)
# model.update_policy(200000, world_model, load_model_step=0)

