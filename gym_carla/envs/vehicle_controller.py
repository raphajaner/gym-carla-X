import numpy as np


def lateral_control(position, velocity, yaw, waypoints):
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

    # Use stanley controller for lateral control
    k_e = 0.001
    k_v = 10

    # 1. calculate heading error between first and last waypoint
    # waypoint = [[x_0, y_0], ..., [x_n, y_n]]
    yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
    print("yaw_path ", yaw_path)
    yaw_diff = yaw_path - yaw
    if yaw_diff > np.pi:
        yaw_diff -= 2 * np.pi
    if yaw_diff < - np.pi:
        yaw_diff += 2 * np.pi

    # 2. calculate crosstrack error
    crosstrack_error = np.min(np.sqrt(np.sum((position - np.array(waypoints)[:, :2]) ** 2, axis=1)))
    print("crosstrack_error ", crosstrack_error)

    yaw_cross_track = np.arctan2(position[1] - waypoints[0][1], position[0] - waypoints[0][0])
    print("yaw_cross_track ", yaw_cross_track)
    #yaw_path2ct = yaw_cross_track - yaw
    yaw_path2ct = yaw_path - yaw_cross_track
    if yaw_path2ct > np.pi:
        yaw_path2ct -= 2 * np.pi
    if yaw_path2ct < - np.pi:
        yaw_path2ct += 2 * np.pi



    if yaw_path2ct < 0:
        crosstrack_error = abs(crosstrack_error)
    else:
        crosstrack_error = - abs(crosstrack_error)

    yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + velocity))
    #yaw_diff_crosstrack=0
    #print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)

    # 3. control low
    steer_expect = yaw_diff + yaw_diff_crosstrack



    if steer_expect > np.pi:
        steer_expect -= 2 * np.pi
    if steer_expect < - np.pi:
        steer_expect += 2 * np.pi

    steer_expect = min(1.22, steer_expect)
    steer_expect = max(-1.22, steer_expect) / 1.22
    print("steer_expect", steer_expect)
    # 4. update
    #steer_output = steer_expect

    #input_steer = (180.0 / np.pi) * steer_output

    # Clamp the steering command to valid bounds
    #steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)

    return steer_expect
