import copy
import time

import carla
import numpy as np
import logging
from queue import Queue
from numpy import random
from sympy import im
from gym_carla.envs.sensors import *
import psutil, os, signal, subprocess

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class CarlaManager:
    def __init__(self, params, verbose=1):
        """ Manages the connection between a Carla server and corresponding client."""
        self.params = copy.deepcopy(params)

        self.client = None
        self.world = None
        self.map = None

        self.traffic_manager = None
        self.spectator = None
        self.vehicle_spawn_points = None
        self.sensor_manager = SensorManager()

        self.settings = None
        self.server_port = 2000 # TODO
        self.tm_port = None
        self.synchronous_mode = None
        self.dt = None

        self.ego = None
        self.ego_bp = None

        # Keeping track of all actors, sensors, etc. in the simulation
        self.vehicles_list = []
        self.walkers_list = []
        self.controller_list = []
        self.sensors_list = []

        self.current_w_frame = None

        # self.init_server(self) TODO
        self.connect_client()

        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.

        # Sensor stuff
        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.params["obs_size"] = int(self.obs_range / self.lidar_bin)
        self.max_ego_spawn_times = params['max_ego_spawn_times']

        # self.dt = None
        # self.settings = self.world.get_settings()
        # self.spectator = self.world.get_spectator()

    def init_server(self):
        # Taken from https://github.com/carla-simulator/rllib-integration
        """Start a server on a random port"""
        self.server_port = random.randint(15000, 32000)

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + self.server_port)
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port + 1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port + 1)

        if self.params["show_display"]:
            server_command = [
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.params["resolution_x"]),
                "-ResY={}".format(self.params["resolution_y"]),
            ]
        else:
            server_command = [
                "DISPLAY= ",
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-opengl"  # no-display isn't supported for Unreal 4.24 with vulkan
            ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.params["quality_level"])
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        # Taken from https://github.com/carla-simulator/rllib-integration
        logging.info('Connecting to Carla server...')
        for i in range(self.params["retries_on_error"]):
            try:
                self.client = carla.Client(self.params["host"], self.server_port)
                self.client.set_timeout(self.params["timeout"])
                self.world = self.client.load_world(self.params['town'])

                # settings = self.world.get_settings()
                # settings.no_rendering_mode = not self.params["enable_rendering"]
                # settings.synchronous_mode = True
                # settings.fixed_delta_seconds = self.params["timestep"]
                # self.world.apply_settings(settings)
                # self.world.tick()
                logging.info('Carla server connected!')
                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1,
                                                                                     self.params["retries_on_error"]))
                time.sleep(3)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def setup_experiment(self):
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.spectator = self.world.get_spectator()
        self._init_traffic_manager()
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        if self.params['weather'] == 'ClearNoon':
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.world.set_pedestrians_cross_factor(self.params['pedestrian_cross_factor'])

    def _init_traffic_manager(self):
        # self.tm_port = self.server_port // 10 + self.server_port % 10
        # while is_used(self.tm_port):
        #    print("Traffic manager's port " + str(self.tm_port) + " is already being used. Checking the next one")
        #    tm_port += 1
        # print("Traffic manager connected to port " + str(self.tm_port))

        # Setup for traffic manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(70.0)
        logging.info('Traffic manager has been created.')

    def tick(self, timeout=10):
        # Send tick to server to move one tick forward, returns the frame number of the snapshot the world will be in
        # after the tick
        self.current_w_frame = self.world.tick()
        logging.debug(f'World frame after tick: {self.current_w_frame}')

        # Move spectator to follow the ego car
        self.set_spectator_camera_view(self.ego.get_transform(), z_offset=5)
        sensors_data = self.sensor_manager.get_data(self.current_w_frame)
        assert all(frame == self.current_w_frame for frame, _ in sensors_data.items())
        return

    def spawn_ego(self):
        self._create_ego_bp()

        self.max_ego_spawn_times = 10
        ego_spawn_times = 0

        if self.ego is not None:
            logging.error("Ego vehicle already exists. Please make sure that the ego is correctly deleted"
                          " before spawning")
            self.ego.destroy()
            self.ego = None

        random.shuffle(self.vehicle_spawn_points)
        for i in range(0,len(self.vehicle_spawn_points)):
            next_spawn_point = self.vehicle_spawn_points[i % len(self.vehicle_spawn_points)]
            self.ego = self.world.try_spawn_actor(self.ego_bp, next_spawn_point)
            if self.ego is not None:
                logging.info("Ego spawned!")
                break
            else:
                logging.warning("Could not spawn hero, changing spawn point")

        if self.ego is None:
            print("We ran out of spawn points")
            # TODO: call self.reset()
            return

        # Spawn sensors to be added to ego vehicle
        #for name, attributes in self.params["sensors"].items():
        camera = RGBCamera('camera', self.params, self.sensor_manager, self.ego)
        lidar = Lidar('camera', self.params, self.sensor_manager, self.ego)
        collision_sensor = CollisionSensor('camera', self.params, self.sensor_manager, self.ego)

        self.sensor_manager.register(camera)
        self.sensor_manager.register(lidar)
        self.sensor_manager.register(collision_sensor)
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
        # if car_lights_on:
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
                # logging.error(f"Spawning walkers lead to error: {result.error}")
                pass
            else:
                # walkers_list.append({"id": result.actor_id})
                walkers_list.append(result.actor_id)
        return walkers_list

    def spawn_pedestrians(self, n_walker):
        # if args.seedw:
        #    world.set_pedestrians_seed(args.seedw)
        #    random.seed(args.seedw)
        walkers_list = []
        # while len(self.walkers_list) < n_walker:
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
                # self.walkers_list[i]["con"] = results[i].actor_id
                controller_list.append(results[i].actor_id)
        # 4. we put together the walkers and controllers id to get the objects from their id
        # for i in range(len(self.walkers_list)):
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
                self.sensor_manager.close()

                if self.ego.destroy():
                    logging.info(f'Destroyed the ego vehicle.')
                else:
                    logging.error(f"Ego vehicle could not be destroyed.")

        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Stop walker controllers (list is [controller, actor, controller, actor ...])
        # if self.walkers_list:
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
        self._sensor_queues = {}

        if self.synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        time.sleep(1)

        logging.info(f"Actors have been destroyed.")

        # self.traffic_manager.global_percentage_speed_difference(30.0)

    def set_spectator_camera_view(self, view=carla.Transform(), z_offset=0):
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




