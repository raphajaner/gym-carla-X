import carla
import numpy as np
from sympy import Point2D, Line2D
import math
from gym_carla.envs.misc import distance_vehicle, distance_vehicle_no_transform_wp


class VehicleController():
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.offset_length = self.calc_front_axle()

    def calc_front_axle(self):
        physics = self.vehicle.get_physics_control()
        wheels = physics.wheels
        # Position returns a Vector3D with measures in cm
        wheel_FL = wheels[0].position
        wheel_FR = wheels[1].position
        wheels_RL = wheels[2].position
        wheels_RR = wheels[3].position
        # Global position of ego in m
        ego_location = self.vehicle.get_transform().location  # Vector3D(x=10, y=20 z=0.5)
        offset_vec_FL = carla.Vector3D(wheel_FL.x / 100, wheel_FL.y / 100, wheel_FL.z / 100).__sub__(ego_location)
        offset_vec_FR = carla.Vector3D(wheel_FR.x / 100, wheel_FR.y / 100, wheel_FR.z / 100).__sub__(ego_location)
        offset_vec = offset_vec_FL + offset_vec_FR
        offset_length = np.sqrt(offset_vec.x ** 2 + offset_vec.y ** 2) / 2
        # waypoints = world.get_map().get_waypoint(vehicle_location, project_to_road=True,
        #                                         lane_type=carla.LaneType.Driving)

        return offset_length

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    def get_front_axle_position(self):
        ego_transform = self.vehicle.get_transform()
        ego_location_x_front_axle = ego_transform.location.x + self.offset_length * math.cos(
            math.radians(ego_transform.rotation.yaw))
        ego_location_y_front_axle = ego_transform.location.y + self.offset_length * math.sin(
            math.radians(ego_transform.rotation.yaw))
        return carla.Transform(carla.Location(ego_location_x_front_axle, ego_location_y_front_axle),
                               ego_transform.rotation)


class LateralVehicleController(VehicleController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.k_e = 3
        self.k_v = 10

        self.yaw_previous = None
        self.wp_target_previous = None

    def lateral_control(self, position, velocity, yaw, waypoints):
        """
            Stanley controller for lateral vehicle control. Derived from:
            https://github.com/diegoavillegasg/Longitudinal-and-Lateral-Controller-on-Carla/blob/master/controller2d.py
                position        : Current [x, y] position in meters
                velocity        : Current forward speed (meters per second)
                yaw             : Current yaw pose in radians
                waypoints       : Current waypoints to track
                                  (Includes speed to track at each x,y
                                  location.)
                                  Format: [[x0, y0, v0],
                                           [x1, y1, v1],
                                           ...
                                           [xn, yn, vn]]
                                  Example:
                                      waypoints[2][1]:
                                      Returns the 3rd waypoint's y position
                                      waypoints[5]:
                                      Returns [x5, y5, v5] (6th waypoint)
        """

        # Change the steer output with the lateral controller.
        steer_output = 0

        print("\n")

        # Get waypoint that is nearest to front wheel
        wp_ind = np.argmin([distance_vehicle_no_transform_wp(wp, self.get_front_axle_position()) for wp in waypoints])
        wp_target = waypoints[wp_ind]
        yaw_path = math.radians(wp_target[2])

        yaw_path = self.normalize_angle(yaw_path)
        yaw = self.normalize_angle(yaw)

        if self.yaw_previous is None:
            self.yaw_previous = yaw

        if self.wp_target_previous is None:
            self.wp_target_previous = wp_target

        # Heading error of car to trajectory
        yaw_diff = (yaw_path - yaw)
        yaw_diff_norm = self.normalize_angle(yaw_diff)

        if abs(yaw_diff_norm) < math.radians(1):
            yaw_diff_norm = 0
        print("yaw_car ", yaw)
        print("yaw_path ", yaw_path)
        print("yaw_diff", yaw_diff)
        print("yaw_diff_norm", yaw_diff_norm)

        # 2. calculate crosstrack error wrt the trajectory
        #crosstrack_error = np.sqrt(np.min(np.sum((position - wp_target[:2]) ** 2)))
        #crosstrack_error2 = np.min(np.linalg.norm(position - wp_target[:2]))
        crosstrack_error = np.min(np.linalg.norm(np.array([self.get_front_axle_position().location.x, self.get_front_axle_position().location.y]) - wp_target[:2]))**2

        print("crosstrack_error_distance ", crosstrack_error)

        front_axle_position = self.get_front_axle_position().location
        front_axle_vec = [front_axle_position.x, front_axle_position.y]
        front_axle_vec_norm = front_axle_vec / np.linalg.norm(front_axle_vec)

        a1_projection = np.dot(wp_target[:2], front_axle_vec_norm) * front_axle_vec_norm

        crosstrack_error = np.linalg.norm(wp_target[:2] - a1_projection)

        print("crosstrack_error_distance norm ", crosstrack_error)

        yaw_cross_track = np.arctan2(wp_target[1] - self.get_front_axle_position().location.y, wp_target[0] - self.get_front_axle_position().location.x)
        # yaw_path2ct = yaw_cross_track - yaw
        print("yaw_cross_track ", yaw_cross_track)
        yaw_cross_track = self.normalize_angle(yaw_cross_track)
        yaw_path2ct = self.normalize_angle(yaw_path - yaw_cross_track)

        print("yaw_path2ct ", yaw_path2ct)

        if yaw_path2ct < 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(self.k_e * crosstrack_error / (self.k_v + velocity))

        print("yaw_diff_crosstrack ", yaw_diff_crosstrack)
        if abs(yaw_diff_crosstrack) < 0.03:
            yaw_diff_crosstrack=0
        # yaw_diff_crosstrack=0
        # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)

        # Yaw daping for extended Stanley control.
        yaw_rate_trajectory = self.normalize_angle(wp_target[2] - self.wp_target_previous[2])
        yaw_rate_ego = self.normalize_angle(yaw - self.yaw_previous)

        yaw_rate_damping = - 0 * (yaw_rate_ego - yaw_rate_trajectory)

        print("yaw_rate_damping ", yaw_rate_damping)


        # 3. control low
        steer_expect = 1.3 * yaw_diff_norm + yaw_diff_crosstrack + yaw_rate_damping

        # if steer_expect > np.pi:
        #    steer_expect -= 2 * np.pi
        # if steer_expect < - np.pi:
        #    steer_expect += 2 * np.pi

        print("steer_expect ", steer_expect)

        # steer_expect = min(1.22, steer_expect)
        # steer_expect = max(-1.22, steer_expect) / 1.22
        print("steer_expect", steer_expect)
        # 4. update
        steer_output = - steer_expect

        input_steer = (180.0 / np.pi) * steer_output / 69.999  # Max steering angle
        print("steerbefore ", input_steer)

        # Clamp the steering command to valid bounds
        # steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        steer = np.clip(input_steer, -1.0, 1.0)
        print("steer ", steer)

        self.yaw_previous = yaw
        self.wp_target_previous = wp_target

        return steer
