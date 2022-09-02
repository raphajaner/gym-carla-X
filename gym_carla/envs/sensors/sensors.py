from abc import ABC, abstractmethod
import numpy as np
import carla
import logging
from queue import Queue, Empty


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
