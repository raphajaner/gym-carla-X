#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import faulthandler; faulthandler.enable()

import time
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] \t %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
from datetime import timedelta
import gym
import gym_carla
def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 196,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.05,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.mercedes.coupe_2020',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town02',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 25,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
        'retries_on_error': 10,
        'timeout': 2,
        'host': '127.0.0.1',
        'weather': 'ClearNoon',
        'pedestrian_cross_factor': 1,
        'ego_spawn_times': 10,
        'sensors': ['RGBCamera', 'Lidar', 'CollisionSensor']
    }
    # Set gym-carla environment
    start_time = time.time()
    env = gym.make('carla-v1', params=params, new_step_api=True)
    blueprint_library = env.world.get_blueprint_library()
    [print(bp.id) for bp in blueprint_library.filter('vehicle.*.*')]
    try:
        env.run(reset_time=30)
    finally:
        logging.info(f'Env run for {str(timedelta(seconds=time.time()-start_time))}s.')
        env.close()

if __name__ == '__main__':
    try:
        main()

    finally:
        logging.info("\nStopped running! Ciao, see you soon!")
