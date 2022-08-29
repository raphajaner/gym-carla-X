import carla
import numpy as np
import logging

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor


class CarlaManager():
    def __init__(self, params, verbose=1):

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        self.client = carla.Client('localhost', params['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(params['town'])
        if verbose > 0:
            print('Carla server connected!')

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





        self.dt = None
        self.settings = self.world.get_settings()
        self.spectator = self.world.get_spectator()
        self.synchronous_mode = None
        self._init_traffic_manager(params)

        # self.actor_list = []
        self.vehicles_list = []
        self.walkers_list = []
        self.sensors_list = []
        self.all_id = []
        self.ego = None


        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        # Set weather
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

        # TODO Make while loop to make sure that there is no vehicle so far!!

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

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

            # if hero:
            #    blueprint.set_attribute('role_name', 'hero')
            #    hero = False
            # else:
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch_vehicle.append(SpawnActor(blueprint, transform)
                                 .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch_vehicle, self.synchronous_mode):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        car_lights_on = False
        if car_lights_on:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, True)

        print("Spawned some vehicles")

    def spawn_pedestrians(self, n_walker):
        bp_walker = [self._create_walker_bp() for _ in range(n_walker)]
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 1.0

        # if args.seedw:
        #    world.set_pedestrians_seed(args.seedw)
        #    random.seed(args.seedw)

        # 1. take all the random locations to spawn
        walker_spawn_points = []
        for i in range(n_walker):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in walker_spawn_points:
            # TODO: Allows to choise the same oint several times!!
            walker_bp = np.random.choice(bp_walker)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (np.random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))

        results = self.client.apply_batch_sync(batch, self.synchronous_mode)

        # for results in self.client.apply_batch_sync(batch, self.synchronous_mode):
        #    if results.error:
        #        logging.error(results.error)
        #    else:
        #        self.vehicles_list.append(results.actor_id)

        # TODO: Only needed to get red of pedestrians that did not get spawned
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))

        results = self.client.apply_batch_sync(batch, self.synchronous_mode)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id

        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])

        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        # print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        print("Spawned walkers")

    def clear_all_actors(self):
        # Delete sensors, vehicles and walkers
        print('\nDestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        if self.ego.is_alive:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()

            self.camera_sensor.stop()
            self.camera_sensor.destroy()

            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()

            self.ego.destroy()


        # Stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\nDestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

    def _init_traffic_manager(self, params, verbose=1):
        # Setup for traffic manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(70.0)
        # self.traffic_manager.global_percentage_speed_difference(30.0)
        if verbose > 0:
            print('Traffic manager created!')

    def set_spectator_view(self, view=carla.Transform()):
        # Get the location and rotation of the spectator through its transform
        # transform = self.spectator.get_transform()
        self.spectator.set_transform(view)

    def set_synchronous_mode(self, params, verbose=1):
        # Set fixed simulation step for synchronous mode
        self.settings.synchronous_mode = True
        self.dt = params['dt']
        self.settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(self.settings)
        self.synchronous_mode = True

    def set_asynchronous_mode(self, verbose=1):
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        self.synchronous_mode = False

    def set_no_rendering_mode(self):
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

    def _create_ego_bp(self):
        self.ego_bp = self._create_vehicle_bp(self.params['ego_vehicle_filter'], color='49,8,8')
        self.ego_bp.set_attribute('role_name', 'hero')
        # self._create_collision_sensor()

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
