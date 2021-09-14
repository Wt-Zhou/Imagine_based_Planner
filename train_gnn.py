import gym
import numpy as np
import gym_routing
import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_03_Follow_Ego,CarEnv_03_Cut_In,CarEnv_03_Cut_In_old
from TestScenario_Town02 import CarEnv_02_Intersection
from Agent.Imagine_Planner import GMBRL
from Agent.world_model.self_attention.self_atten_world_model import GNN_World_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = CarEnv_03_Cut_In_old()
world_model = GNN_World_Model(obs_shape=env.observation_space.shape,
            action_shape=[1], # discrete, 1 dimension!
            state_space_dim=env.state_dimension,
            device=device,
            env=env)
world_model.train_world_model(env, load_step=20000, train_step=30000)
# world_model.try_inference(env, load_step=20000)

# model = GMBRL(world_model)
# model.learn_dqn(100000, env, load_model_step=0)

