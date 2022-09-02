from abc import ABC, abstractmethod
import numpy as np
import carla
import logging
from queue import Queue, Empty
from gym_carla.envs.sensors.sensors import *

class SensorManager:
    def __init__(self):
        self._sensors = []
        self._sensor_queues = {}  # Queue()  # {name: Sensor object}
        # self._data_buffers = #Queue()
        self._queue_timeout = 10
        self._sensor_timeout = 2

        self._event_sensor_queues = {}  # Queue()
        # self._event_data_buffers = Queue()

    @property
    def sensors(self):
        #sensors = {name: sensor for name, sensor in self._sensor_queues.items()}
        #sensors.update({name: sensor for name, sensor in self._event_sensor_queues.items()})
        return self._sensors

    def spawn_ego_sensors(self, ego):
        # Spawn sensors to be added to ego vehicle
        # for name, attributes in self.params["sensors"].items():
        camera = RGBCamera('camera', self.params, self.sensor_manager, self.ego)
        lidar = Lidar('lidar', self.params, self.sensor_manager, self.ego)
        collision_sensor = CollisionSensor('collision_sensor', self.params, self.sensor_manager, self.ego)

        self.register(camera)
        self.register(lidar)
        self.register(collision_sensor)

    def reset(self):
        for sensor in self.sensors:
            sensor.close()
        self._sensor_queues = {}
        self._event_sensor_queues = {}#Queue()

        # self._data_buffers = Queue()
        # self._event_data_buffers = Queue()

    def close(self):
        self.reset()
        #self._sensor_queues = None
        #self._event_sensor_queues = None
        # self._data_buffers = None
        # self._event_data_buffers = None

    def register(self, sensor):
        """Adds a specific sensor to the class"""
        self._sensors.append(sensor)
        if sensor.is_event_sensor:
            self._event_sensor_queues[sensor.name] = sensor.data_queue
        else:
            self._sensor_queues[sensor.name] = sensor.data_queue

    def get_data(self, w_frame):
        data_all = {}
        if not self._sensor_queues:
            for sensor_name, sensor_queue in self._sensor_queues.items():
                while True:
                    # Ensure that data is always from the same world frame w_frame
                    frame, data = sensor_queue.get(timeout=2)
                    if frame == w_frame:
                        logging.debug(f"Queue {sensor_name}: {frame} == {w_frame}")
                        return data
                    else:
                        logging.warning(f"Queue {sensor_name}: {frame} != {w_frame}")
                data_all.update({sensor_name: (frame, data)})

        if not self._event_sensor_queues:
            for sensor_name, sensor_queue in self._event_sensor_queues.items():
                while True:
                    try:
                        # Ensure that data is always from the same world frame w_frame
                        frame, data = sensor_queue.get_nowait()
                        if frame == w_frame:
                            logging.debug(f"Queue {sensor_name}: {frame} == {w_frame}")
                            return data
                        else:
                            logging.warning(f"Queue {sensor_name}: {frame} != {w_frame}")
                    except Empty:
                        frame = w_frame
                        data = None
                data_all.update({sensor_name: (frame, data)})
        return data_all

