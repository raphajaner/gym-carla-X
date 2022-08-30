import time

import carla
import numpy as np
import logging

from sympy import im

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor


class CarlaManager():
    def __init__(self, params, verbose=1):

        # Connect to carla server and get world object
        print('Connecting to Carla server...')
        self.client = carla.Client('localhost', params['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(params['town'])
        if verbose > 0:
            print('Carla server connected!')

        self.params = params
        self.dt = params['dt']
        self.task_mode = params['task_mode']

        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.max_ego_spawn_times = params['max_ego_spawn_times']

        self.dt = None
        self.settings = self.world.get_settings()
        self.spectator = self.world.get_spectator()
        self.synchronous_mode = None
        self._init_traffic_manager(params)

        self.all_actors_list = []
        self.ego = None
        self.vehicles_list = []
        self.walkers_list = []
        self.controller_list = []
        self.sensors_list = []
        self.all_id = []

        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.world.set_pedestrians_cross_factor(1)

    def spawn_ego(self):

        self._create_ego_bp()
        self._create_collision_sensor_bp()
        self._create_lidar_bp()
        self._create_camera_bp()

        self.max_ego_spawn_times = 10
        self.task_mode = 'random'
        # Spawn the ego vehicle
        ego_spawn_times = 0
        vehicle = None

        # while True:
        #     if ego_spawn_times > self.max_ego_spawn_times:
        #        self.reset()

        transform = np.random.choice(self.vehicle_spawn_points)
        self.ego = self.world.try_spawn_actor(self.ego_bp, transform)
        if self.ego is not None:
            logging.info(f"Ego vehicles spawned.")
        else:
            logging.error(f"Ego vehicles could not be spawned.")

        # TODO Make while loop to make sure that there is no vehicle so far!!

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)

        # self.collision_sensor.listen(lambda event: get_collision_hist(event))



        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)

        # self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)

        # self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        return self.ego

    def spawn_vehicle(self, n_vehicles):
        batch_vehicle = []
        blueprints = [self._create_vehicle_bp() for _ in range(n_vehicles)]

        for n, transform in enumerate(self.vehicle_spawn_points):
            if n >= n_vehicles:
                break
            blueprint = np.random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            blueprint.set_attribute('role_name', 'autopilot')

            # TODO: should be a function
            traffic_manager_get_port = 8000

            # spawn the cars and set their autopilot and light state all together
            batch_vehicle.append(SpawnActor(blueprint, transform)
                                 .then(SetAutopilot(FutureActor, True, traffic_manager_get_port)))

        response = self.client.apply_batch_sync(batch_vehicle, self.synchronous_mode)
        for results in response:
            if results.error:
                logging.error(f"Spawning vehicles lead to error: {results.error}")
            else:
                self.vehicles_list.append(results.actor_id)

        self.all_actors_list += self.vehicles_list

        # Set automatic vehicle lights update if specified
        car_lights_on = False
        #if car_lights_on:
        #    all_vehicle_actors = self.world.get_actors(self.vehicles_list)
        #    for actor in all_vehicle_actors:
        #        self.traffic_manager.update_vehicle_lights(actor, True)

        logging.info(f'Spawned {len(self.vehicles_list)} vehicles.')

    def _spawn_batch_pedestrians(self, n_walker):

        blueprints = [self._create_walker_bp() for _ in range(n_walker)]

        # 1. Take all the random locations to spawn
        walker_spawn_points = []
        for _ in range(n_walker):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        # walker_speed = []
        np.random.shuffle(blueprints)

        walkers_list = []
        batch = [SpawnActor(bp, sp) for (bp, sp) in zip(blueprints, walker_spawn_points)]

        response = self.client.apply_batch_sync(batch, self.synchronous_mode)

        for result in response:
            if result.error:
                logging.error(f"Spawning walkers lead to error: {result.error}")

            else:
                #walkers_list.append({"id": result.actor_id})
                walkers_list.append(result.actor_id)
        return walkers_list

    def spawn_pedestrians(self, n_walker):
        # if args.seedw:
        #    world.set_pedestrians_seed(args.seedw)
        #    random.seed(args.seedw)
        walkers_list = []
        #while len(self.walkers_list) < n_walker:
        #    try:
        walkers_list += self._spawn_batch_pedestrians(n_walker=n_walker - len(walkers_list))
        #    except:
        #        logging.error("Error when spawning walkers.")

        self.walkers_list = walkers_list
        self.all_actors_list += walkers_list

        # Spawn the walker controller
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker_id) for walker_id in walkers_list]
        results = self.client.apply_batch_sync(batch, self.synchronous_mode)

        controller_list = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(f"Error when spawning the controllers: {results[i].error}")
            else:
                #self.walkers_list[i]["con"] = results[i].actor_id
                controller_list.append(results[i].actor_id)
        # 4. we put together the walkers and controllers id to get the objects from their id
        #for i in range(len(self.walkers_list)):
        #    self.all_id.append(self.walkers_list[i]["con"])
        #    self.all_id.append(self.walkers_list[i]["id"])
        self.controller_list = controller_list
        self.all_actors_list += controller_list

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        actor_controller_list = self.world.get_actors(self.controller_list)

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        for actor in actor_controller_list:
            # start walker
            actor.start()
            # set walk to random point
            actor.go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            # self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        logging.info(f'Spawned {len(self.walkers_list)} walkers.')

    def clear_all_actors(self):

        # Delete sensors, vehicles and walkers
        # due_tick_cue=False
        response = self.client.apply_batch_sync([DestroyActor(x) for x in self.vehicles_list])
        if response:
            n_deleted = 0
            for result in response:
                if result.error:
                    logging.error(f"A vehicle could not be destroyed: {result.error}")
                else:
                    n_deleted += 1
            logging.info(f'Destroyed {n_deleted} vehicle')
        else:
            logging.error(f"There were no vehicles to be destroyed.")


        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if self.ego is not None:
            if self.ego.is_alive:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()

                self.camera_sensor.stop()
                self.camera_sensor.destroy()

                self.lidar_sensor.stop()
                self.lidar_sensor.destroy()

                if self.ego.destroy():
                    logging.info(f'Destroyed the ego vehicle.')
                else:
                    logging.error(f"Ego vehicle could not be destroyed.")

        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Stop walker controllers (list is [controller, actor, controller, actor ...])
        #if self.walkers_list:
        #    for i in range(0, len(self.all_id), 2):
        #        self.all_actors[i].stop()

        for controller in self.world.get_actors(self.controller_list):
            controller.stop()

        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()


        response = self.client.apply_batch_sync([DestroyActor(x) for x in self.controller_list])

        if response:
            n_deleted = 0
            for result in response:
                if result.error:
                    logging.error(f"A controller could not be destroyed: {result.error}")
                else:
                    n_deleted += 1
            logging.info(f'Destroyed {n_deleted} controller')
        else:
            logging.error(f"There were no controller to be destroyed.")

        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        response = self.client.apply_batch_sync([DestroyActor(x) for x in self.walkers_list])

        if response:
            n_deleted = 0
            for result in response:
                if result.error:
                    logging.error(f"A walker could not be destroyed: {result.error}")
                else:
                    n_deleted += 1
            logging.info(f'Destroyed {n_deleted} walkers')
        else:
            logging.error(f"There were no walker to be destroyed.")

        self.all_actors_list = []
        self.ego = None
        self.vehicles_list = []
        self.walkers_list = []
        self.controller_list = []
        self.sensors_list = []
        self.all_id = []


        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        logging.info(f"Actors have been destroyed.")
        time.sleep(1)

    def _init_traffic_manager(self, params, ):
        # Setup for traffic manager
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
        logging.info('Traffic manager created.')

        return traffic_manager
        # self.traffic_manager.global_percentage_speed_difference(30.0)

    def set_spectator_view(self, view=carla.Transform(), z_offset=0):
        # Get the location and rotation of the spectator through its transform
        # transform = self.spectator.get_transform()
        view.location.z += z_offset
        vec = view.rotation.get_forward_vector()
        view.location.x -= 2 * vec.x
        view.location.y -= 2 * vec.y

        view.rotation.pitch = -20
        self.spectator.set_transform(view)

    def set_synchronous_mode(self, params):
        # Set fixed simulation step for synchronous mode
        self.settings.synchronous_mode = True
        self.dt = params['dt']
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)
        self.synchronous_mode = True

    def set_asynchronous_mode(self):
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        self.synchronous_mode = False
        logging.warning('Carla simulation set to asynchronous mode.')

    def set_no_rendering_mode(self):
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)
        logging.warning('Carla simulation set to no rendering mode.')

    def _create_ego_bp(self):
        self.ego_bp = self._create_vehicle_bp(self.params['ego_vehicle_filter'], color='49,8,8')
        self.ego_bp.set_attribute('role_name', 'hero')

    def _create_vehicle_bp(self, ego_vehicle_filter='vehicle.*', color=None):
        """Returns:
        bp: the blueprint object of carla.
        """

        blueprints = self.world.get_blueprint_library().filter(ego_vehicle_filter)
        blueprint_library = []

        for nw in [4]:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = np.random.choice(blueprint_library)

        if bp.has_attribute('color'):
            if not color:
                color = np.random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        return bp

    def _create_walker_bp(self):
        walker_bp = np.random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')

        # walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        return walker_bp  # , walker_controller_bp

    def _create_collision_sensor_bp(self):
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Lidar sensor

    def _create_lidar_bp(self):
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '5000')

    def _create_camera_bp(self):
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')
