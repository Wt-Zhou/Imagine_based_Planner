import gym
import numpy as np
import gym_routing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_03_Cut_In
from Agent.CP import CP
from Agent.CP import Imagine_Model


env = CarEnv_03_Cut_In()
model = CP()
corner_buffer = model.collect_corner_data(env)
imagine_model = Imagine_Model(corner_buffer)

empty_env = CarEnv_03_Cut_In(spawn_env_veh = False)

# Train or load world model
# imagine_model.update_transition_and_reward(empty_env)
load_step = 620000
imagine_model.load_world_model(empty_env, load_step)
model.learn(20000, imagine_model, load_model_step=0)

