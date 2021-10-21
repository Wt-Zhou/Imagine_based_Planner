import gym
import numpy as np
import gym_routing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario_handcontrol import CarEnv_03_HandControl
from Agent.CP_New import CP
from Agent.CP_New import Log_Replay_Imagine_Model_New


# 1. Init Model

# --Policy Init
env = CarEnv_03_HandControl(spawn_env_veh = True, handcontrol = True)
model = CP()
# --Train or Load Ego AV Dynamics
# empty_env = CarEnv_03_HandControl(spawn_env_veh = False, handcontrol = False)
imagine_model = Log_Replay_Imagine_Model_New(env)
# imagine_model.train_ego_dynamics(empty_env, 20)
# load_step = 20000
# imagine_model.load_ego_dynamics(empty_env, load_step)


for i in range(0, 100):
    
    # 2. Drive, Collect Data
    # 3. Corner Case Data Collection
    imagine_model.collect_corner_data(env)

    # 4. Imagine and Improve Policy
    model.learn(10, imagine_model, load_model_step=0)


    # 5. Test Policy with Corner Case
    # model.test_corner_case(imagine_model)





