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
        self.carla_manager = CarlaManager(params)

        # parameters
        self.params = params
        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        self.task_mode = params['task_mode']
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt']
        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route']
        if 'pixor' in params.keys():
            self.pixor = params['pixor']
            self.pixor_size = params['pixor_size']
        else:
            self.pixor = False

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
        if self.pixor:
            observation_space_dict.update({
                'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
                'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
                'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]),
                                          dtype=np.float32)
            })
        self.observation_space = spaces.Dict(observation_space_dict)



        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

    @property
    def world(self):
        return self.carla_manager.world

    @property
    def client(self):
        return self.carla_manager.client

    def reset(self, seed=None, return_info=True, options=None):
        logging.info("Env gym-carla will be reset.")
        #pygame.quit()

        self.carla_manager.clear_all_actors()

        # Initialize the renderer

        #import time
        time.sleep(1)
        #self.carla_manager.client.reload_world(reset_settings=False)
        #self.carla_manager.world = self.carla_manager.client.get_world()
        #Reload the current world, note that a        new
        #world is created with default settings using the same map.All actors present in the
        #world will be destroyed, but traffic manager instances will stay alive.


        # Disable sync mode
        self.carla_manager.set_asynchronous_mode()
        self.ego = self.carla_manager.spawn_ego()
        self.carla_manager.spawn_vehicle(5)
        self.carla_manager.spawn_pedestrians(89)
        # Enable sync mode
        self.carla_manager.set_synchronous_mode(self.params)

        #self._init_renderer()

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1



        self.ego_controller = LateralVehicleController(self.ego)
        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()


        # Set ego information for render
        #self.birdeye_render.set_hero(self.ego, self.ego.id)

        # Set spectator view to ego vehicle
        self.carla_manager.set_spectator_view(self.ego.get_transform())
        return_info = {}
        #return ({'birdeye':None, 'camera':None, 'lidar':None, 'state':None}, return_info) #self._get_obs()
        return ({'birdeye':None, 'camera':None, 'lidar':None, 'state':None}, return_info) #self._get_obs()

    def step(self, action):


        # Calculate acceleration and steering

        ego_trans = self.ego.get_transform()

        self.carla_manager.set_spectator_view(ego_trans, z_offset=3)


        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = math.radians(ego_trans.rotation.yaw)
        ego_velocity = self.ego.get_velocity()
        ego_velocity = np.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2)

        a_steer = self.ego_controller.lateral_control(np.array([ego_x, ego_y]), ego_velocity, ego_yaw, self.waypoints)


        action = np.array([action[0], a_steer])

        # For each Wheel Physics Control, print maximum steer anglewwsasdwa
        physics_control = self.ego.get_physics_control()
        #for wheel in physics_control.wheels:
        #    print(wheel.max_steer_angle)

        # measurements, sensor_data = self.client.read_data()

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
        self.world.tick()

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # state information
        # info = {
        #    'waypoints': self.waypoints,
        #    'vehicle_front': self.vehicle_front
        # }
        info = {}

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        #return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))
        #return (None, self._get_reward(), self._terminal(), copy.deepcopy(info))
        # New api obs, reward, termination, truncation, info.
        #return (self._get_obs(), 1, False, False, copy.deepcopy(info))
        return ({'birdeye':None, 'camera':None, 'lidar':None, 'state':None}, 1, False, False, copy.deepcopy(info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def run(self, verbose=0, reset_time=20):
        obs, _ = self.reset(return_info=True)
        start_time = time.time()
        while True:
            t1 = time.time()
            action = [1.3, 0.0]
            obs, r, terminated, truncated, info = self.step(action)
            t2 = time.time()
            if t2 - start_time > reset_time:
                truncated = True
                start_time = t2
            if verbose > 0:
                logging.info(f"Step takes {t2 - t1}")
            if terminated:
                obs, _ = self.reset()
            if truncated:
                obs, _ = self.reset()

    def close(self):
        self.carla_manager.clear_all_actors()
        pygame.quit()
        settings = self.world.get_settings()
        #settings.synchronous_mode = False
        #settings.no_rendering_mode = False
        #settings.fixed_delta_seconds = None
        #self.world.apply_settings(settings)
        print('Cleaned all actors successfully!')
        super().close()


    def _init_renderer(self):
        """Initialize the birdeye view renderer.
    """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(True, 8000)
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

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

    def _get_obs(self):
        """Get the observations."""
        ## Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints

        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        # Roadmap
        if self.pixor:
            roadmap_render_types = ['roadmap']
            if self.display_route:
                roadmap_render_types.append('waypoints')
            self.birdeye_render.render(self.display, roadmap_render_types)
            roadmap = pygame.surfarray.array3d(self.display)
            roadmap = roadmap[0:self.display_size, :, :]
            roadmap = display_to_rgb(roadmap, self.obs_size)
            # Add ego vehicle
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    if abs(birdeye[i, j, 0] - 255) < 20 and abs(birdeye[i, j, 1] - 0) < 20 and abs(
                            birdeye[i, j, 0] - 255) < 20:
                        roadmap[i, j, :] = birdeye[i, j, :]

        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        self.display.blit(birdeye_surface, (0, 0))

        ## Lidar image generation
        point_cloud = []
        # Get point cloud data
        for location in self.carla_manager.lidar_data:
            point_cloud.append([location.point.x, location.point.y, -location.point.z])
        point_cloud = np.array(point_cloud)
        # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
        # and z is set to be two bins.
        y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
        x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
        z_bins = [-self.carla_manager.lidar_height - 1, -self.carla_manager.lidar_height + 0.25, 1]
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
        lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
        # Add the waypoints to lidar image
        if self.display_route:
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

        # Display lidar image
        lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        self.display.blit(lidar_surface, (self.display_size, 0))

        ## Display camera image
        if self.carla_manager.camera_img.shape[0] != self.obs_size:
            camera = resize(self.carla_manager.camera_img, (self.obs_size, self.obs_size)) * 255
        else:
            camera = self.carla_manager.camera_img.astype(np.float)
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size * 2, 0))

        # Display on pygame
        pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w,
                                       np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

        if self.pixor:
            ## Vehicle classification and regression maps (requires further normalization)
            vh_clas = np.zeros((self.pixor_size, self.pixor_size))
            vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

            # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
            # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
            for actor in self.world.get_actors().filter('vehicle.*'):
                x, y, yaw, l, w = get_info(actor)
                x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
                if actor.id != self.ego.id:
                    if abs(y_local) < self.obs_range / 2 + 1 and x_local < self.obs_range - self.d_behind + 1 and x_local > -self.d_behind - 1:
                        x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
                            local_info=(x_local, y_local, yaw_local, l, w),
                            d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
                        cos_t = np.cos(yaw_pixel)
                        sin_t = np.sin(yaw_pixel)
                        logw = np.log(w_pixel)
                        logl = np.log(l_pixel)
                        pixels = get_pixels_inside_vehicle(
                            pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
                            pixel_grid=self.pixel_grid)
                        for pixel in pixels:
                            vh_clas[pixel[0], pixel[1]] = 1
                            dx = x_pixel - pixel[0]
                            dy = y_pixel - pixel[1]
                            vh_regr[pixel[0], pixel[1], :] = np.array(
                                [cos_t, sin_t, dx, dy, logw, logl])

            # Flip the image matrix so that the origin is at the left-bottom
            vh_clas = np.flip(vh_clas, axis=0)
            vh_regr = np.flip(vh_regr, axis=0)

            # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
            pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

        obs = {
            'camera': camera.astype(np.uint8),
            'lidar': lidar.astype(np.uint8),
            'birdeye': birdeye.astype(np.uint8),
            'state': state,
        }

        if self.pixor:
            obs.update({
                'roadmap': roadmap.astype(np.uint8),
                'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
                'vh_regr': vh_regr.astype(np.float32),
                'pixor_state': pixor_state,
            })

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        r_speed = -abs(speed - self.desired_speed)

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
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
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
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            return True

        return False

