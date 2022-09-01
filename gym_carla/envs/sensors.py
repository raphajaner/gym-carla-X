from abc import ABC, abstractmethod
import numpy as np
import carla
import logging
from queue import Queue, Empty


class SensorManager:
    def __init__(self):
        self._sensors = []
        self._sensors_queue = {}  # Queue()  # {name: Sensor object}
        # self._data_buffers = #Queue()
        self._queue_timeout = 10
        self._sensor_timeout = 2

        self._event_sensors_queue = {}  # Queue()
        # self._event_data_buffers = Queue()

    @property
    def sensors(self):
        #sensors = {name: sensor for name, sensor in self._sensors_queue.items()}
        #sensors.update({name: sensor for name, sensor in self._event_sensors_queue.items()})
        return self._sensors

    def reset(self):
        for sensor in self.sensors:
            sensor.close()
        self._sensors_queue = {}
        self._event_sensors_queue = {}#Queue()

        # self._data_buffers = Queue()
        # self._event_data_buffers = Queue()

    def close(self):
        self.reset()
        #self._sensors_queue = None
        #self._event_sensors_queue = None
        # self._data_buffers = None
        # self._event_data_buffers = None

    def register(self, sensor):
        """Adds a specific sensor to the class"""
        self._sensors.append(sensor)
        if sensor.is_event_sensor:
            self._event_sensors_queue[sensor.name] = sensor.data_queue
        else:
            self._sensors_queue[sensor.name] = sensor.data_queue

    def get_data(self, w_frame):
        data_all = {}
        if not self._sensors_queue:
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

        if not self._event_sensors_queue:
            for sensor_name, sensor_queue in self._event_sensors_queue.items():
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


class Sensor(ABC):
    def __init__(self, name, params, manager, parent, bp, transform):
        # Use a queue for thread safe access to the data
        self.data_queue = Queue()

        self.name = name
        self.params = params
        self.manager = manager
        self.parent = parent
        self.bp = bp
        self.transform = transform

        # TODO: What is best to setup a abstract class with arguments that have to be added later?!
        self.is_event_sensor = None

        self.create_bp()

        world = self.parent.get_world()
        self.sensor = world.spawn_actor(self.bp, self.transform, attach_to=self.parent)
        self.sensor.listen(self.callback)

    @property
    def id(self):
        return self.sensor.id

    def parse(self, data):
        return data

    def update(self, frame, data):
        # Update the data queue
        data_processed = self.parse(data)
        self.data_queue.put((frame, data_processed))

    #@abstractmethod
    def create_bp(self):
        pass

    def callback(self, data):
        # The callback is wrapping the update function to allow the use in the sensor.listen function (otherwise, a
        # lambda function had to be used)
        self.update(data.frame, data)

    def close(self):
        if self.sensor is not None:
            if self.sensor.is_listening:
                self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


class EventSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_event_sensor = True
        self.last_event_frame = None


class ContinuousSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_event_sensor = False


class RGBCamera(ContinuousSensor):
    def __init__(self, name, params, manager, parent):
        self.world = parent.get_world()

        self.obs_size = params['obs_size']
        bp, transform = self.create_bp()
        super().__init__(name, params, manager, parent, bp, transform)

    def create_bp(self):
        camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')

        # Modify the attributes of the blueprint to set image resolution and field of view.
        bp.set_attribute('image_size_x', str(self.obs_size))
        bp.set_attribute('image_size_y', str(self.obs_size))
        bp.set_attribute('fov', '110')

        # Set the time in seconds between sensor captures
        # TODO: Should I keep this?!
        bp.set_attribute('sensor_tick', '0.02')
        return bp, transform

    def parse(self, data):
        data_processed = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        data_processed = np.reshape(data_processed, (data.height, data.width, 4))
        data_processed = data_processed[:, :, :3]
        return data_processed[:, :, ::-1]


class Lidar(ContinuousSensor):
    def __init__(self, name, params, manager, parent):
        self.world = parent.get_world()
        self.obs_size = params['obs_size']

        bp, transform = self.create_bp()
        super().__init__(name, params, manager, parent, bp, transform)

    def create_bp(self):
        lidar_height = 2.1
        transform = carla.Transform(carla.Location(x=0.0, z=lidar_height))
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', '32')
        bp.set_attribute('range', '5000')
        return bp, transform


class CollisionSensor(EventSensor):
    def __init__(self, name, params, manager, parent):
        self.world = parent.get_world()

        bp, transform = self.create_bp()
        super().__init__(name, params, manager, parent, bp, transform)

    def create_bp(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        return bp, carla.Transform()

    def parse(self, data):
        impulse = data.normal_impulse
        intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # self.collision_hist.append(intensity)
        # if len(self.collision_hist) > self.collision_hist_l:
        #    self.collision_hist.pop(0)
        return intensity

    def callback(self, data):
        # The collision sensor can have multiple callbacks per tick. Get only the first one
        if self.last_event_frame != data.frame:
            self.last_event_frame = data.frame
            self.update(data.frame, data)

        # logging.info(f"collision added {data.frame}")
        # if not queue.empty():
        #    old_data = queue.get()
        #    if old_data[0] != data.frame:
        #        queue.put(old_data)
