import gym
import numpy as np
import gym_routing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_03_Cut_In
from Agent.CP import CP


env = CarEnv_03_Cut_In()
model = CP()

# # Different Training Steps
steps_each_test = 1

model_list = [50019,100083,150036]
# for i in range(0,20000,500):
#     model_list.append(i)
# model_list = [0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800]

model.test(model_list=model_list, test_steps=steps_each_test, env=env)



print("Finish UBP test")
