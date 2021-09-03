import glob
import os
import sys
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass
try:
	sys.path.append(glob.glob("/home/icv/.local/lib/python3.6/site-packages/")[0])
except IndexError:
	pass

import carla
import time
import numpy as np
import math
import random
import cv2
import gym

from tqdm import tqdm
from collections import deque
from gym import core, error, spaces, utils
from gym.utils import seeding
from TestScenario import CarEnv_03_Cut_In
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.CP import CP, Imagine_Model
EPISODES=2642

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_03_Cut_In()

    # Create Agent
    trajectory_planner = JunctionTrajectoryPlanner()
    controller = Controller()
    dynamic_map = DynamicMap()
    target_speed = 30/3.6 

    pass_time = 0
    task_time = 0

    # Loop over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        obs = env.reset()
        episode_reward = 0

        done = False

        decision_count = 0

        # Loop over steps
        while True:
            obs = np.array(obs)
            dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, action = trajectory_planner.trajectory_update(dynamic_map)

            action = np.array(1)
            print("action",action)
            # Control
            rule_trajectory = trajectory_planner.trajectory_update_CP(action, rule_trajectory)

            for i in range(5):
                control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
                action = [control_action.acc, control_action.steering]
                new_obs, reward, done, _ = env.step(action)   
                dynamic_map.update_map_from_obs(new_obs, env)
                if done:
                    break
                # Set current step for next loop iteration
            obs = new_obs
            episode_reward += reward

            

            if done:
                trajectory_planner.clear_buff()
                task_time += 1
                if reward > 0:
                    pass_time += 1
                break

        print("Episode Reward:",episode_reward)
        print("Success Rate:",pass_time/task_time)
        

