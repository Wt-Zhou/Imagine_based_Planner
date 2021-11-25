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


import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_0
from pygame.locals import K_9
from pygame.locals import K_BACKQUOTE
from pygame.locals import K_BACKSPACE
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_b
from pygame.locals import K_c
from pygame.locals import K_d
from pygame.locals import K_g
from pygame.locals import K_h
from pygame.locals import K_i
from pygame.locals import K_l
from pygame.locals import K_m
from pygame.locals import K_n
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_v
from pygame.locals import K_w
from pygame.locals import K_x
from pygame.locals import K_z
from pygame.locals import K_MINUS
from pygame.locals import K_EQUALS
from Agent.CARLA_manual_control import HUD, KeyboardControl
from Agent.CARLA_manual_control_steeringwheel import HUD, DualControl

MAP_NAME = 'Town03'
OBSTACLES_CONSIDERED = 1 # For cut-in in Town03

global goal_point_03
goal_point_03 = Transform()
goal_point_03.location.x = 245
goal_point_03.location.y = 26
goal_point_03.location.z = 0
goal_point_03.rotation.pitch = 0
goal_point_03.rotation.yaw = -90 
goal_point_03.rotation.roll = 0

global start_point_03_2
start_point_03_2 = Transform()
start_point_03_2.location.x = 242
start_point_03_2.location.y = 150
start_point_03_2.location.z = 2
start_point_03_2.rotation.pitch = 0
start_point_03_2.rotation.yaw = -90
start_point_03_2.rotation.roll = 0

class CarEnv_03_HandControl:

    def __init__(self, spawn_env_vehicle = True, handcontrol = False, equipment = 1):

        self.spawn_env_vehicle = spawn_env_vehicle
        # CARLA settings
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        if self.world.get_map().name != MAP_NAME:
            self.world = self.client.load_world(MAP_NAME)
        # self.world.set_weather(carla.WeatherParameters(cloudiness=0, precipitation=10.0, sun_altitude_angle=90.0))
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = False
        self.settings.fixed_delta_seconds = 0.02 # Warning: When change simulator, the delta_t in controller should also be change.
        # settings.substepping = True
        # settings.max_substep_delta_time = 0.02  # fixed_delta_seconds <= max_substep_delta_time * max_substeps
        # settings.max_substeps = 10
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)
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
        self.action_space = spaces.Discrete(7) # len(fplist + 1) 0 for brake 
        self.state_dimension = 10
        self.low  = np.array([245,  80, -5, -5,-5,245,  80, -5, -5,-5], dtype=np.float64)
        self.high = np.array([250,  100, 1, 1,5, 250,  100, 1, 1,5], dtype=np.float64)    
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)

        # Set Ego Vehicle
        self.ego_vehicle_collision_sign = False
        self.stuck_time = 0

        # Control Env Vehicle
        self.has_set = np.zeros(1000000)
        self.stopped_time = np.zeros(1000000)   

        # Record
        self.log_dir = "record.txt"
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

        # Case
        # self.init_case()
        self.case_id = 0


        # Spawn env vehicle

        # if self.spawn_env_veh:
        #     self.spawn_fixed_veh()

        # Corner Case
        self.handcontrol = handcontrol
        if self.handcontrol and self.spawn_env_vehicle:
            self.spawn_env_veh()
            
            pygame.init()
            pygame.font.init()
            display = pygame.display.set_mode(
            (320, 200),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

            display.fill((0,0,0))
            pygame.display.flip()

            # hud = HUD(320, 200)

            # world = World(client.get_world(), hud, args)
            self.keyboard_controller = DualControl(self.env_vehicle, equipment = equipment) #equipment:1-keyboard, 2-steeringwheel, 3-remote steeringwheel

            self.keyboard_clock = pygame.time.Clock()
       
    def free_traffic_lights(self, carla_world):
        traffic_lights = carla_world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(5)
            tl.set_red_time(5)

    def global_routing(self):
        global goal_point_03
        global start_point_03_2

        start = start_point_03_2
        goal = goal_point_03
        # print("Calculating route to x={}, y={}, z={}".format(
        #         goal.location.x,
        #         goal.location.y,
        #         goal.location.z))
        
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
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

    def ego_vehicle_stuck(self, stay_thres = 5):        
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)
        if ego_vehicle_velocity < 0.1:
            pass
        else:
            self.stuck_time = time.time()

        if time.time() - self.stuck_time > stay_thres:
            return True
        return False

    def ego_vehicle_pass(self):
        global goal_point_03
        ego_location = self.ego_vehicle.get_location()
        if ego_location.distance(goal_point_03.location) < 30:
            return True
        else:
            return False

    def ego_vehicle_collision(self, event):
        self.ego_vehicle_collision_sign = True

    def wrap_state(self):
        # state = [0 for i in range((OBSTACLES_CONSIDERED + 1) * 4)]
        state  = np.array([0,  0, 0, 0,0,245,  80, -5, -5,-5], dtype=np.float64)

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
        state[4] = ego_vehicle_state.yaw 

        # Obs state
        closest_obs = []
        closest_obs = self.found_closest_obstacles()
        i = 0
        for obs in closest_obs: 
            if i < OBSTACLES_CONSIDERED:
                if obs[0] != 0:
                    state[(i+1)*5+0] = obs[0] #- ego_ffstate.s 
                    state[(i+1)*5+1] = obs[1] #+ ego_ffstate.d
                    state[(i+1)*5+2] = obs[2]
                    state[(i+1)*5+3] = obs[3]
                    state[(i+1)*5+4] = obs[4]
                i = i+1
            else:
                break
        
        return state

    def found_closest_obstacles(self):
        obs_tuples = []
        for obs in self.world.get_actors().filter('vehicle*'): 
            # Calculate distance
            p1 = np.array([self.ego_vehicle.get_location().x ,  self.ego_vehicle.get_location().y])
            p2 = np.array([obs.get_location().x , obs.get_location().y])
            p3 = p2 - p1
            p4 = math.hypot(p3[0],p3[1])
            
            # Obstacles too far
            one_obs = (obs.get_location().x, obs.get_location().y, obs.get_velocity().x, obs.get_velocity().y, obs.get_transform().rotation.yaw / 180.0 * math.pi , p4, obs.get_transform().rotation.yaw)
            if p4 > 0:
                obs_tuples.append(one_obs)
        
        closest_obs = []
        fake_obs = [0 for i in range(11)]  #len(one_obs)
        for i in range(0, OBSTACLES_CONSIDERED ,1): # 3 obs
            closest_obs.append(fake_obs)
        
        # Sort by distance
        sorted_obs = sorted(obs_tuples, key=lambda obs: obs[5])   

        for obs in sorted_obs:
            closest_obs[0] = obs 

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
            # print("[CARLA]: Record To Txt: All", self.task_num, self.stuck_num, self.collision_num, self.case_id )

    def clean_task_nums(self):
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

    def reset(self):    
        # Control Env Elements
        if self.spawn_env_vehicle:
            self.spawn_env_veh()

        self.spawn_ego_veh()
        self.ego_vehicle_collision_sign = False

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
        throttle = max(0,float(action[0]))  # range [0,1]
        brake = max(0,-float(action[0])) # range [0,1]
        steer = action[1] # range [-1,1]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake, steer = steer))
        self.world.tick()

        # State
        state = self.wrap_state()

        # Step reward
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)

        reward = ego_vehicle_velocity / 15 

        # Hand control of env vehicle
        if self.handcontrol and self.spawn_env_vehicle:
            self.keyboard_control()

        # If finish
        done = False
        if self.ego_vehicle_collision_sign:
            self.collision_num += + 1
            done = True
            print("[CARLA]: Collision!")
        
        if self.ego_vehicle_pass():
            done = True
            print("[CARLA]: Successful!")

        elif self.ego_vehicle_stuck():
            self.stuck_num += 1
            done = True
            print("[CARLA]: Stuck!")


        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        for vehicle in vehicle_list:  
            self.tm.ignore_signs_percentage(vehicle, 100)
            self.tm.ignore_lights_percentage(vehicle, 100)
            self.tm.ignore_walkers_percentage(vehicle, 0)
            self.tm.set_percentage_keep_right_rule(vehicle,100) # it can make the actor go forward, dont know why
            self.tm.global_percentage_speed_difference(-100) 
            self.tm.auto_lane_change(vehicle, True)
            self.tm.distance_to_leading_vehicle(vehicle, 10)
            self.tm.collision_detection(vehicle, self.ego_vehicle, True)

        return state, reward, done, None

    def step_replay(self, action, env_veh_trans):
        # Control ego vehicle
        throttle = max(0,float(action[0]))  # range [0,1]
        brake = max(0,-float(action[0])) # range [0,1]
        steer = action[1] # range [-1,1]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake, steer = steer))
        self.world.tick()

        # State
        state = self.wrap_state()

        # Step reward
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)

        reward = ego_vehicle_velocity / 15 

        # Hand control of env vehicle
        if self.handcontrol and self.spawn_env_vehicle:
            self.keyboard_control()
        elif env_veh_trans == -1:
            self.env_vehicle.set_autopilot()
            self.tm.auto_lane_change(self.env_vehicle, True)
            self.tm.ignore_signs_percentage(self.env_vehicle, 100)
            self.tm.ignore_lights_percentage(self.env_vehicle, 100)
            
        elif not self.handcontrol and env_veh_trans.location.z >= -0.1:
            env_veh_trans.location.z = self.ego_vehicle.get_location().z
            env_veh_trans.rotation.pitch = self.ego_vehicle.get_transform().rotation.pitch
            env_veh_trans.rotation.roll = self.ego_vehicle.get_transform().rotation.roll

            self.env_vehicle.set_transform(env_veh_trans)
            self.env_vehicle.set_target_velocity(carla.Vector3D(0,0,0))
        
        # If finish
        done = False
        if self.ego_vehicle_collision_sign:
            self.collision_num += + 1
            done = True
            # print("[CARLA]: Collision!")
        
        if self.ego_vehicle_pass():
            done = True
            # print("[CARLA]: Successful!")

        elif self.ego_vehicle_stuck():
            self.stuck_num += 1
            done = True
            # print("[CARLA]: Stuck!")

        return state, reward, done, None

    def spawn_env_veh(self):
        try:
            self.env_vehicle.set_autopilot(enabled=False)
            self.env_vehicle.destroy()
        except:
            pass

        transform = Transform()
        transform.location.x = 235 
        transform.location.y = 130 
        transform.location.z = 2
        transform.rotation.pitch = 0
        transform.rotation.yaw = -90
        transform.rotation.roll = 0

        blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.mercedes-benz.coupe'))
        if blueprint.has_attribute('color'):
            color = '0,0,255' # blue
            blueprint.set_attribute('color', color)
        
        self.env_vehicle = self.world.spawn_actor(blueprint, transform)
        
    def spawn_ego_veh(self):
        global start_point_03_2
        try:
            self.ego_collision_sensor.destroy()
            self.ego_vehicle.destroy()
        except:
            pass

        self.ego_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz2017'))
        if self.ego_vehicle_bp.has_attribute('color'):
            color = '255,0,0'
            self.ego_vehicle_bp.set_attribute('color', color)
            self.ego_vehicle_bp.set_attribute('role_name', "ego_vehicle")
        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, start_point_03_2)
        # self.ego_vehicle.set_target_velocity(carla.Vector3D(0,-10,0))

        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.ego_collision_sensor = self.world.spawn_actor(collision_bp, Transform(), self.ego_vehicle, carla.AttachmentType.Rigid)
        self.ego_collision_sensor.listen(lambda event: self.ego_vehicle_collision(event))

    def keyboard_control(self):            
        self.keyboard_clock.tick_busy_loop(60)
        self.keyboard_controller.parse_events(self.env_vehicle, self.keyboard_clock)
        # world.tick(clock)
        # world.render(display)
        pygame.event.pump()
        pygame.display.flip()
        return 0


