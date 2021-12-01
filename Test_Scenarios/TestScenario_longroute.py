import glob
import os
import sys
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    # sys.path.append(glob.glob("/home/icv/.local/lib/python3.6/site-packages/")[0])
except IndexError:
	pass

import carla
import time
import numpy as np
import math
import random
import gym
import cv2
import threading
from random import randint
from carla import Location, Rotation, Transform, Vector3D, VehicleControl
from collections import deque
from tqdm import tqdm
from gym import core, error, spaces, utils
from gym.utils import seeding
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from Agent.zzz.dynamic_map import Lanepoint, Lane, Vehicle
from Agent.zzz.tools import *

global start_point_02
start_point_02 = Transform()
start_point_02.location.x = -43
start_point_02.location.y = 183
start_point_02.location.z = 1
start_point_02.rotation.pitch = 0
start_point_02.rotation.yaw = -220
start_point_02.rotation.roll = 0


global goal_point_02
goal_point_02 = Transform()
goal_point_02.location.x = -77
goal_point_02.location.y = 150
goal_point_02.location.z = 0
goal_point_02.rotation.pitch = 0
goal_point_02.rotation.yaw = 0 
goal_point_02.rotation.roll = 0

class CarEnv_10_Long_Test:

    def __init__(self):
        
        # CARLA settings
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        if self.world.get_map().name != 'Carla/Maps/Town10':
            self.world = self.client.load_world('Town10')
        self.world.set_weather(carla.WeatherParameters(cloudiness=50, precipitation=10.0, sun_altitude_angle=30.0))
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = 0.1 # Warning: When change simulator, the delta_t in controller should also be change.
        settings.substepping = True
        settings.max_substep_delta_time = 0.02  # fixed_delta_seconds <= max_substep_delta_time * max_substeps
        settings.max_substeps = 10
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.free_traffic_lights(self.world)

        self.tm = self.client.get_trafficmanager(8000)
        # self.tm.set_hybrid_physics_mode(True)
        # self.tm.set_hybrid_physics_radius(50)
        self.tm.set_random_device_seed(0)

        actors = self.world.get_actors().filter('vehicle*')
        for actor in actors:
            actor.destroy()

        # Generate Reference Path
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1)
        self.global_routing()

        # RL settingss
        self.action_space = spaces.Discrete(8) # len(fplist + 2) 0 for rule, 1 for brake 
        self.low  = np.array([-70,  160, -5, -1,-70,  160, -5, -1, -70,  160, -5, -1, -70,  160, -5, -1], dtype=np.float64)
        self.high = np.array([-30,  190, 1, 1,  -30,  190, 1, 1, -30,  190, 1, 1, -30,  190, 1, 1], dtype=np.float64)    
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # Spawn Ego Vehicle
        global start_point_02
        # self.ego_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.mercedes-benz.coupe'))
        self.ego_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz_2020'))
        if self.ego_vehicle_bp.has_attribute('color'):
            color = '255,0,0'
            self.ego_vehicle_bp.set_attribute('color', color)
            self.ego_vehicle_bp.set_attribute('role_name', "ego_vehicle")
        self.ego_collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.ego_vehicle = None
        self.stuck_time = 0
        
        # self.env_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.mercedes-benz.coupe'))
        self.env_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.tt'))
        if self.env_vehicle_bp.has_attribute('color'):
            color = '0,0,255'
            self.env_vehicle_bp.set_attribute('color', color)
        if self.env_vehicle_bp.has_attribute('driver_id'):
            driver_id = random.choice(self.env_vehicle_bp.get_attribute('driver_id').recommended_values)
            self.env_vehicle_bp.set_attribute('driver_id', driver_id)
            self.env_vehicle_bp.set_attribute('role_name', 'autopilot')

        # Control Env Vehicle
        self.has_set = np.zeros(1000000)
        self.stopped_time = np.zeros(1000000)   

        # Record
        self.log_dir = "record.txt"
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

        # Case
        self.init_case()
        self.case_id = 0
       
    def free_traffic_lights(self, carla_world):
        traffic_lights = carla_world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(5)
            tl.set_red_time(5)

    def global_routing(self):
        global goal_point_02
        global start_point_02

        start = start_point_02
        goal = goal_point_02
        print("Calculating route to x={}, y={}, z={}".format(
                goal.location.x,
                goal.location.y,
                goal.location.z))
        
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1)
        # grp = GlobalRoutePlanner(dao) # Carla 0911
        # grp.setup()

        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1) # Carla 0913
        current_route = grp.trace_route(carla.Location(start.location.x,
                                                start.location.y,
                                                start.location.z),
                                carla.Location(goal.location.x,
                                                goal.location.y,
                                                goal.location.z))
        t_array = []
        self.ref_path = Lane()
        for wp in current_route:
            lanepoint = Lanepoint()
            lanepoint.position.x = wp[0].transform.location.x 
            lanepoint.position.y = wp[0].transform.location.y
            self.ref_path.central_path.append(lanepoint)
            t_array.append(lanepoint)
        self.ref_path.central_path_array = np.array(t_array)
        self.ref_path.speed_limit = 60/3.6 # m/s

        ref_path_ori = convert_path_to_ndarray(self.ref_path.central_path)
        self.ref_path_array = dense_polyline2d(ref_path_ori, 2)
        self.ref_path_tangets = np.zeros(len(self.ref_path_array))

    def ego_vehicle_stuck(self, stay_thres = 2):        
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)
        if ego_vehicle_velocity < 0.1:
            pass
        else:
            self.stuck_time = time.time()

        if time.time() - self.stuck_time > stay_thres:
            return True
        
    def ego_vehicle_pass(self):
        global goal_point_02
        ego_location = self.ego_vehicle.get_location()
        if ego_location.distance(goal_point_02.location) < 20:
            return True
        else:
            return False

    def ego_vehicle_collision(self, event):
        self.ego_vehicle_collision_sign = True

    def wrap_state(self):
        # state = [0 for i in range((OBSTACLES_CONSIDERED + 1) * 4)]
        state  = np.array([-41,  180, -0, -0,-15, 190, 0, 0, -15, 190, 0, 0, -15, 190, 0, 0], dtype=np.float64)

        ego_vehicle_state = Vehicle()
        ego_vehicle_state.x = self.ego_vehicle.get_location().x
        ego_vehicle_state.y = self.ego_vehicle.get_location().y
        ego_vehicle_state.v = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)

        ego_vehicle_state.yaw = self.ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad
        ego_vehicle_state.yawdt = self.ego_vehicle.get_angular_velocity()

        ego_vehicle_state.vx = ego_vehicle_state.v * math.cos(ego_vehicle_state.yaw)
        ego_vehicle_state.vy = ego_vehicle_state.v * math.sin(ego_vehicle_state.yaw)

        # Ego state
        state[0] = ego_vehicle_state.x 
        state[1] = ego_vehicle_state.y 
        state[2] = ego_vehicle_state.vx 
        state[3] = ego_vehicle_state.vy 

        # Obs state
        closest_obs = []
        closest_obs = self.found_closest_obstacles_ramp()
        i = 0
        for obs in closest_obs: 
            if i < OBSTACLES_CONSIDERED:
                if obs[0] != 0:
                    state[(i+1)*4+0] = obs[0] 
                    state[(i+1)*4+1] = obs[1] 
                    state[(i+1)*4+2] = obs[2]
                    state[(i+1)*4+3] = obs[3]
                i = i+1
            else:
                break
        
        return state

    def found_closest_obstacles_ramp(self):
        obs_tuples = []
        for obs in self.world.get_actors().filter('vehicle*'): 
            # Calculate distance
            p1 = np.array([self.ego_vehicle.get_location().x ,  self.ego_vehicle.get_location().y])
            p2 = np.array([obs.get_location().x , obs.get_location().y])
            p3 = p2 - p1
            p4 = math.hypot(p3[0],p3[1])
            
            # Obstacles too far
            one_obs = (obs.get_location().x, obs.get_location().y, obs.get_velocity().x, obs.get_velocity().y, p4, obs.get_transform().rotation.yaw)
            if p4 > 0:
                obs_tuples.append(one_obs)
        
        closest_obs = []
        fake_obs = [0 for i in range(11)]  #len(one_obs)
        for i in range(0, OBSTACLES_CONSIDERED ,1): # 3 obs
            closest_obs.append(fake_obs)
        
        # Sort by distance
        sorted_obs = sorted(obs_tuples, key=lambda obs: obs[4])   

        put_1st = False
        put_2nd = False
        put_3rd = False
        for obs in sorted_obs:
            if obs[0] > -50 and obs[1] < 200 and put_1st == False:
                closest_obs[0] = obs 
                put_1st = True
                continue
            if obs[0] < -50 and obs[1] < 200 and put_2nd == False:
                closest_obs[1] = obs
                put_2nd = True
                continue
            if obs[0] < -50 and obs[1] < 200 and put_3rd == False:
                closest_obs[2] = obs
                put_3rd = True
                continue
            else:
                continue
        return closest_obs
                                            
    def record_information_txt(self):
        if self.task_num > 0:
            stuck_rate = float(self.stuck_num) / float(self.task_num)
            collision_rate = float(self.collision_num) / float(self.task_num)
            pass_rate = 1 - ((float(self.collision_num) + float(self.stuck_num)) / float(self.task_num))
            fw = open(self.log_dir, 'a')   
            # Write num
            fw.write(str(self.task_num)) 
            fw.write(", ")
            fw.write(str(self.case_id)) 
            fw.write(", ")
            fw.write(str(self.stuck_num)) 
            fw.write(", ")
            fw.write(str(self.collision_num)) 
            fw.write(", ")
            fw.write(str(stuck_rate)) 
            fw.write(", ")
            fw.write(str(collision_rate)) 
            fw.write(", ")
            fw.write(str(pass_rate)) 
            fw.write("\n")
            fw.close()               
            print("[CARLA]: Record To Txt: All", self.task_num, self.stuck_num, self.collision_num, self.case_id )

    def clean_task_nums(self):
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

    def reset(self):    

        # Env vehicles
        self.spawn_fixed_veh()

        # Ego vehicle
        self.spawn_ego_veh()
        self.world.tick() 


        # State
        state = self.wrap_state()

        # Record
        self.record_information_txt()
        self.task_num += 1
        self.case_id += 1

        return state

    def step(self, action):
        # Control ego vehicle
        # print("action[0]",action[0])
        throttle = max(0,float(action[0]))  # range [0,1]
        brake = max(0,-float(action[0])) # range [0,1]
        steer = action[1] # range [-1,1]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake, steer = steer))
        self.world.tick()

        # State
        state = self.wrap_state()

        # Step reward
        reward = 0
        # If finish
        done = False
        if self.ego_vehicle_collision_sign:
            self.collision_num += + 1
            done = True
            reward = 0
            print("[CARLA]: Collision!")
        
        if self.ego_vehicle_pass():
            done = True
            reward = 1
            print("[CARLA]: Successful!")

        elif self.ego_vehicle_stuck():
            self.stuck_num += 1
            reward = -0.0
            done = True
            print("[CARLA]: Stuck!")

        return state, reward, done, None

    def init_case(self):
        self.case_list = []
        
        useless_transform = Transform()
        useless_transform.location.x = -45 
        useless_transform.location.y = 204
        useless_transform.location.z = 1
        useless_transform.rotation.pitch = 0
        useless_transform.rotation.yaw = 0
        useless_transform.rotation.roll = 0
        
        useless_transform_2 = Transform()
        useless_transform_2.location.x = -65 
        useless_transform_2.location.y = 200
        useless_transform_2.location.z = 1
        useless_transform_2.rotation.pitch = 0
        useless_transform_2.rotation.yaw = 0
        useless_transform_2.rotation.roll = 0
        
        # one vehicle behind
        for i in range(0,10):
            spawn_vehicles = []
            transform = Transform()
            transform.location.x = -45 + i * 5
            transform.location.y = 193
            transform.location.z = 1
            transform.rotation.pitch = 0
            transform.rotation.yaw = -180
            transform.rotation.roll = 0
            spawn_vehicles.append(transform)
            spawn_vehicles.append(useless_transform)
            spawn_vehicles.append(useless_transform_2)
            self.case_list.append(spawn_vehicles)

        # one vehicle front
        for i in range(0,5):
            spawn_vehicles = []
            transform = Transform()
            transform.location.x = -59 + i * 1 
            transform.location.y = 188
            transform.location.z = 1
            transform.rotation.pitch = 0
            transform.rotation.yaw = -140
            transform.rotation.roll = 0
            spawn_vehicles.append(transform)
            spawn_vehicles.append(useless_transform)
            spawn_vehicles.append(useless_transform_2)
            self.case_list.append(spawn_vehicles)

        # two vehicles
        for i in range(0,10):
            for j in range(0,10):
                spawn_vehicles = []
                transform = Transform()
                transform.location.x = -45 + i * 5
                transform.location.y = 193
                transform.location.z = 1
                transform.rotation.pitch = 0
                transform.rotation.yaw = -180
                transform.rotation.roll = 0
                spawn_vehicles.append(transform)
                transform = Transform()
                transform.location.x = -52 + i * 5
                transform.location.y = 196
                transform.location.z = 12
                transform.rotation.pitch = 0
                transform.rotation.yaw = -180
                transform.rotation.roll = 0
                spawn_vehicles.append(transform)
                spawn_vehicles.append(useless_transform)
                spawn_vehicles.append(useless_transform_2)
                self.case_list.append(spawn_vehicles)

        # 3 vehicles
        for i in range(0,10):
            for j in range(0,10):
                for k in range (0,5):
                    spawn_vehicles = []
                    transform = Transform()
                    transform.location.x = -45 + i * 5
                    transform.location.y = 193
                    transform.location.z = 1
                    transform.rotation.pitch = 0
                    transform.rotation.yaw = -180
                    transform.rotation.roll = 0
                    spawn_vehicles.append(transform)
                    
                    transform = Transform()
                    transform.location.x = -52 + i * 5
                    transform.location.y = 196
                    transform.location.z = 1
                    transform.rotation.pitch = 0
                    transform.rotation.yaw = -180
                    transform.rotation.roll = 0
                    spawn_vehicles.append(transform)
                    
                    transform = Transform()
                    transform.location.x = -59 + i * 1 
                    transform.location.y = 188
                    transform.location.z = 1
                    transform.rotation.pitch = 0
                    transform.rotation.yaw = -140
                    transform.rotation.roll = 0
                    spawn_vehicles.append(transform)
                    spawn_vehicles.append(useless_transform)
                    spawn_vehicles.append(useless_transform_2)
                    self.case_list.append(spawn_vehicles)

        print("How many Cases?",len(self.case_list))

    def spawn_fixed_veh(self):
        if self.case_id >= len(self.case_list):
            self.case_id = 1
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        synchronous_master = True

        for vehicle in vehicle_list:
            if vehicle.attributes['role_name'] != "ego_vehicle" :
                vehicle.destroy()

        batch = []
        print("Case_id",self.case_id)

        for transform in self.case_list[self.case_id - 1]:
            batch.append(SpawnActor(self.env_vehicle_bp, transform).then(SetAutopilot(FutureActor, True)))
    
        self.client.apply_batch_sync(batch, synchronous_master)

        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        for vehicle in vehicle_list:  
            self.tm.ignore_signs_percentage(vehicle, 100)
            self.tm.ignore_lights_percentage(vehicle, 100)
            self.tm.ignore_walkers_percentage(vehicle, 0)
            self.tm.auto_lane_change(vehicle, True)
            # self.tm.distance_to_leading_vehicle(vehicle, 10)

    def spawn_ego_veh(self):
        global start_point_02
        if self.ego_vehicle is not None:
            self.ego_collision_sensor.destroy()
            self.ego_vehicle.destroy()

        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, start_point_02)
        self.ego_collision_sensor = self.world.spawn_actor(self.ego_collision_bp, Transform(), self.ego_vehicle, carla.AttachmentType.Rigid)
        self.ego_collision_sensor.listen(lambda event: self.ego_vehicle_collision(event))
        self.ego_vehicle_collision_sign = False
