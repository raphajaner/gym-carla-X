#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import time

import gym
import gym_carla
import carla


def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 196,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town02',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 15,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
    }
    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)

    try:
        obs = env.reset()
        print("lllloooop")
        while True:
            action = [1.3, 0.0]
            t = time.time()
            obs, r, done, info = env.step(action)
            print("step takes", time.time()-t)
            #print("lllloooop")
            #if done:
            #    obs = env.reset()
    finally:
        env._clear_all_actors(['sensor.*', 'vehicle.*',
                               'controller.ai.walker', 'walker.*'])
        settings = env.world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        env.world.apply_settings(settings)

        print('Cleaned all actors successfully!')


if __name__ == '__main__':
    main()
