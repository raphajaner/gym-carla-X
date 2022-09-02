from gym_carla.envs.sensors.sensors import *


class SensorManager:
    def __init__(self, params, client):
        self.params = params
        self.client = client
        self.world = self.client.get_world()

        self._sensors = {}
        self._sensor_queues = {}  # Queue()  # {name: Sensor object}
        # self._data_buffers = #Queue()
        self._queue_timeout = 10
        self._sensor_timeout = 2

        self._event_sensor_queues = {}  # Queue()
        # self._event_data_buffers = Queue()

    @property
    def sensors(self):
        # sensors = {name: sensor for name, sensor in self._sensor_queues.items()}
        # sensors.update({name: sensor for name, sensor in self._event_sensor_queues.items()})
        return self._sensors

    def spawn_ego_sensors(self, ego):
        # Spawn sensors to be added to ego vehicle
        # for name, attributes in self.params["sensors"].items():

        for sensor_typ in self.params['sensors']:
            if sensor_typ == 'RGBCamera':
                sensor = RGBCamera('camera', self.params, self, ego)
            elif sensor_typ == 'Lidar':
                sensor = Lidar('lidar', self.params, self, ego)
            elif sensor_typ == 'CollisionSensor':
                sensor = CollisionSensor('collision_sensor', self.params, self, ego)
            else:
                sensor = None
            if sensor is not None:
                self.register(sensor)

    def close_all_sensors(self):
        for sensor in self.sensors.values():
            sensor.close()
        self._sensor_queues = {}
        self._event_sensor_queues = {}

    def register(self, sensor):
        """Adds a specific sensor to the class"""
        self._sensors.update({sensor.name: sensor})
        if sensor.is_event_sensor:
            self._event_sensor_queues[sensor.name] = sensor.data_queue
        else:
            self._sensor_queues[sensor.name] = sensor.data_queue

    def get_data(self, w_frame):
        data_all = {}
        if self._sensor_queues:
            for sensor_name, sensor_queue in self._sensor_queues.items():
                while True:
                    # Ensure that data is always from the same world frame w_frame
                    frame, data = sensor_queue.get(timeout=2)
                    if frame == w_frame:
                        logging.debug(f"Queue {sensor_name}: {frame} == {w_frame}")
                        break
                    else:
                        logging.warning(f"Queue {sensor_name}: {frame} != {w_frame}")
                data_all.update({sensor_name: (frame, data)})

        if self._event_sensor_queues:
            for sensor_name, sensor_queue in self._event_sensor_queues.items():
                while True:
                    try:
                        # Ensure that data is always from the same world frame w_frame
                        frame, data = sensor_queue.get_nowait()
                        if frame == w_frame:
                            logging.debug(f"Queue {sensor_name}: {frame} == {w_frame}")
                            break
                        else:
                            logging.warning(f"Queue {sensor_name}: {frame} != {w_frame}")
                    except Empty:
                        frame = w_frame
                        data = None
                        break
                data_all.update({sensor_name: (frame, data)})
        return data_all
