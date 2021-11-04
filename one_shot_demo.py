import gym
import numpy as np
# import gym_routing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Test_Scenarios.TestScenario_handcontrol import CarEnv_03_HandControl
from Agent.CP_New import CP
from Agent.CP_New import Log_Replay_Imagine_Model_New


# 1. Init Model

# --Policy Init
model = CP()
# --Train or Load Ego AV Dynamics
empty_env = CarEnv_03_HandControl(spawn_env_vehicle = False, handcontrol = False)
imagine_model = Log_Replay_Imagine_Model_New(empty_env)
# imagine_model.train_ego_dynamics(empty_env, 300000)
load_step = 350000
imagine_model.load_ego_dynamics(empty_env, load_step)

# for i in range(0, 100):
    
# 2. Drive, Collect Data
# 3. Corner Case Data Collection
hand_control_env = CarEnv_03_HandControl(spawn_env_vehicle = True, handcontrol = True)
imagine_model.collect_corner_data(hand_control_env)

# 4. Imagine and Improve Policy
model.learn(5000, imagine_model, load_model_step=0)


# 5. Test Policy with Corner Case
test_env = CarEnv_03_HandControl(spawn_env_vehicle = True, handcontrol = False)
model.test_corner_case(test_env, imagine_model)





