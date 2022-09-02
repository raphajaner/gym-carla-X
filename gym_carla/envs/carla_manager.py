import copy
import time

import numpy as np
from numpy import random
from gym_carla.envs.sensors.sensor_manager import SensorManager
from gym_carla.envs.actors.actor_manager import ActorManager
import psutil, os, signal, subprocess
import logging
import carla


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
        self.spectator = None

        self.settings = None
        self.server_port = 2000  # TODO
        self.tm_port = None
        self.synchronous_mode = None
        self.dt = None

        self.traffic_manager = None
        self.sensor_manager = None
        self.actor_manager = None

        self.current_w_frame = None

        self.setup_experiment()

    def setup_experiment(self):
        # self.init_server(self) TODO
        self.connect_client()

        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        if self.params['carla_no_rendering']:
            self.set_no_rendering_mode()
        self.spectator = self.world.get_spectator()

        self.init_traffic_manager()

        self.actor_manager = ActorManager(self.params, self.client)
        self.sensor_manager = SensorManager(self.params, self.client)

        if self.params['weather'] == 'ClearNoon':
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.world.set_pedestrians_cross_factor(self.params['pedestrian_cross_factor'])

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

    def init_traffic_manager(self):
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
        logging.info(f'Port of the traffic manager is {self.traffic_manager.get_port()}.')
        logging.info(f'Traffic manager has been created.')

    def tick(self, timeout=10):
        # Send tick to server to move one tick forward, returns the frame number of the snapshot the world will be in
        # after the tick
        self.current_w_frame = self.world.tick()
        logging.debug(f'World frame after tick: {self.current_w_frame}')

        # Move spectator to follow the ego car
        sensors_data = self.sensor_manager.get_data(self.current_w_frame)
        assert all(frame == self.current_w_frame for frame, _ in sensors_data.values())
        return sensors_data

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

    def clear_all_actors(self):
        self.sensor_manager.close_all_sensors()
        self.actor_manager.clear_all_actors()

        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
