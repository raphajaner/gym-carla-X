#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import logging

import numpy as np
import pygame
import random
import time
from datetime import timedelta

from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner, RoadOption
from gym_carla.envs.misc import *
from gym_carla.envs.vehicle_controller import LateralVehicleController
from gym_carla.envs.carla_manager import CarlaManager


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        self.time_start = time.time()

        self.params = params
        self.obs_size = int(self.params['obs_range'] / self.params['lidar_bin'])
        self.carla_manager = CarlaManager(params)
        self.carla_manager.set_synchronous_mode(self.params)


        # action and observation spaces
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'], params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
                                                     params['continuous_steer_range'][0]]),
                                           np.array([params['continuous_accel_range'][1],
                                                     params['continuous_steer_range'][1]]),
                                           dtype=np.float32)  # acc, steer
        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }

        self.observation_space = spaces.Dict(observation_space_dict)

        # print(self.carla_manager.settings.synchronous_mode)
        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0


    @property
    def world(self):
        return self.carla_manager.world

    @property
    def client(self):
        return self.carla_manager.client

    @property
    def ego(self):
        return self.carla_manager.actor_manager.ego

    def reset(self, seed=None, return_info=True, options=None):
        logging.info("Env gym-carla will be reset.")

        self.carla_manager.clear_all_actors()
        # Disable sync mode for spawning
        self.carla_manager.set_asynchronous_mode()
        self.carla_manager.actor_manager.spawn_vehicles(self.params['number_of_vehicles'])
        self.carla_manager.actor_manager.spawn_walkers(self.params['number_of_walkers'])
        self.carla_manager.actor_manager.spawn_ego(self.params)
        self.carla_manager.sensor_manager.spawn_ego_sensors(self.ego)
        self.carla_manager.set_synchronous_mode(self.params)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        self._init_renderer()
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        self.ego_controller = LateralVehicleController(self.ego)
        self.routeplanner = RoutePlanner(self.ego, self.params['max_waypt'])
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Set spectator view to ego vehicle
        if self.params['follow_cam_ego']:
            self.carla_manager.set_spectator_camera_view(self.ego.get_transform())

        self.carla_manager.current_w_frame = self.carla_manager.world.get_snapshot().frame
        data = self.carla_manager.sensor_manager.get_data(self.carla_manager.current_w_frame)

        return_info = {}
        return self._get_obs(data), return_info  # self._get_obs()

    def step(self, action):

        snapshot = self.carla_manager.world.get_snapshot()
        logging.info(f'Snapshot in step has frame {snapshot.frame}')

        ego_snapshot = snapshot.find(self.ego.id)
        # Calculate acceleration and steering
        ego_trans = ego_snapshot.get_transform()

        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = math.radians(ego_trans.rotation.yaw)
        ego_velocity = self.ego.get_velocity()
        # Do not use ego_velocity.length() since this includes the z axis
        ego_velocity = np.linalg.norm([ego_velocity.x, ego_velocity.y])

        a_steer = self.ego_controller.lateral_control(np.array([ego_x, ego_y]), ego_velocity, ego_yaw, self.waypoints)
        action = np.array([action[0], a_steer])

        if ego_velocity > 50 / 3.6:
            action[0] = 0
        if self.discrete:
            acc = self.discrete_act[0][action // self.n_steer]
            steer = self.discrete_act[1][action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)
        # self.world.debug.draw_point(self.ego_controller.get_front_axle_position().location, size=0.5)

        data = self.carla_manager.tick()
        if self.params['follow_cam_ego']:
            self.carla_manager.set_spectator_camera_view(self.ego.get_transform(), z_offset=5)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # State information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return self._get_obs(data), 1, False, False, copy.deepcopy(info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def run(self, verbose=0, reset_time=20):
        obs, _ = self.reset(return_info=True)
        start_time = time.time()
        while True:
            action = [1.3, 0.0]
            t1 = time.time()
            obs, r, terminated, truncated, info = self.step(action)
            t2 = time.time()
            logging.info(f"Loop run for {str(timedelta(seconds=t2 - t1))}")
            duration_loop = t2 - start_time

            if duration_loop > reset_time:
                truncated = True
                start_time = t2

            if terminated:
                obs, _ = self.reset()
                logging.info(f"Loop run for {str(timedelta(seconds=duration_loop))}")
            if truncated:
                obs, _ = self.reset()
                logging.info(f"Loop run for {str(timedelta(seconds=duration_loop))}")

    def close(self):
        self.carla_manager.clear_all_actors()
        pygame.quit()
        super().close()
        logging.info('Gym carla env has been closed!')

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
    """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.params['display_size'] * 3, self.params['display_size']),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.params['display_size'] / self.params['obs_range']
        pixels_ahead_vehicle = (self.params['obs_range'] / 2 - self.params['d_behind']) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.params['display_size'], self.params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self, data):
        obs = {}

        """Get the observations."""
        # Birdeye rendering
        self.birdeye_render.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.birdeye_render.walker_polygons = [self._get_actor_polygons('walker.*')]
        self.birdeye_render.waypoints = self.waypoints

        # Birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap']
        if self.params['display_route']:
            birdeye_render_types.append('waypoints')

        self.birdeye_render.render(self.display, birdeye_render_types)

        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.params['display_size'], :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        obs.update({'birdeye': birdeye.astype(np.uint8)})

        # Display birdeye image
        if self.params['display_rendering']:
            birdeye_surface = rgb_to_display_surface(birdeye, self.params['display_size'])
            self.display.blit(birdeye_surface, (0, 0))

        # Lidar image generation

        if 'Lidar' in self.params['sensors']:
            point_cloud = []
            lidar_height = self.carla_manager.sensor_manager.sensors['lidar'].lidar_height
            # Get point cloud data
            for location in data['lidar'][1]:
                point_cloud.append([location.point.x, location.point.y, -location.point.z])
            point_cloud = np.array(point_cloud)
            # Separate the 3D space to bins for point cloud, x and y is set according to self.params['lidar_bin'],
            # and z is set to be two bins.
            y_bins = np.arange(-(self.params['obs_range'] - self.params['d_behind']),
                               self.params['d_behind'] + self.params['lidar_bin'], self.params['lidar_bin'])
            x_bins = np.arange(-self.params['obs_range'] / 2, self.params['obs_range'] / 2 + self.params['lidar_bin'],
                               self.params['lidar_bin'])
            z_bins = [-lidar_height - 1, -lidar_height + 0.25, 1]
            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
            lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
            # Add the waypoints to lidar image
            if self.params['display_route']:
                wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
            else:
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(np.rot90(wayptimg, 3))

            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)
            lidar = np.flip(lidar, axis=1)
            lidar = np.rot90(lidar, 1)
            lidar = lidar * 255

            obs.update({'lidar': lidar.astype(np.uint8)})

            # Display lidar image
            if self.params['display_rendering']:
                lidar_surface = rgb_to_display_surface(lidar, self.params['display_size'])
                self.display.blit(lidar_surface, (self.params['display_size'], 0))

        if 'RGBCamera' in self.params['sensors']:
            ## Display camera image
            camera_img = data['camera'][1]  # self.carla_manager.sensor_manager.sensors['camera'].camera_img

            if camera_img.shape[0] != self.obs_size:
                camera = resize(camera_img, (self.obs_size, self.obs_size)) * 255
            else:
                camera = camera_img.astype(np.float64)
            if self.params['display_rendering']:
                camera_surface = rgb_to_display_surface(camera, self.params['display_size'])
                self.display.blit(camera_surface, (self.params['display_size'] * 2, 0))

            obs.update({'camera': camera.astype(np.uint8)})

            # Display on pygame
            if self.params['display_rendering']:
                pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])
        obs.update({'state': state})

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        r_speed = -abs(speed - self.params['desired_speed'])

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.params['out_lane_thres']:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.params['desired_speed']:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon ** 2

        r = 200 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep
        if self.time_step > self.params['max_time_episode']:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.params['out_lane_thres']:
            return True

        return False
