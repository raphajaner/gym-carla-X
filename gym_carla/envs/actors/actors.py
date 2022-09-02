import numpy as np


def create_ego_bp(params, world):
    ego_bp = create_vehicle_bp(world, params['ego_vehicle_filter'], color='49,8,8')
    ego_bp.set_attribute('role_name', 'hero')
    return ego_bp


def create_vehicle_bp(world, ego_vehicle_filter='vehicle.*', color=None):
    """Returns:
    bp: the blueprint object of carla.
    """

    blueprints = world.get_blueprint_library().filter(ego_vehicle_filter)

    blueprint_library = []

    for nw in [4]:
        blueprint_library = blueprint_library + [x for x in blueprints if
                                                 int(x.get_attribute('number_of_wheels')) == nw]

    vehicle_bp = np.random.choice(blueprint_library)

    if vehicle_bp.has_attribute('color'):
        if not color:
            color = np.random.choice(vehicle_bp.get_attribute('color').recommended_values)
        vehicle_bp.set_attribute('color', color)
    vehicle_bp.set_attribute('role_name', 'autopilot')

    return vehicle_bp


def create_walker_bp(world):
    walker_bp = np.random.choice(world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
        walker_bp.set_attribute('is_invincible', 'true')

    # walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
    return walker_bp  # , walker_controller_bp


def create_walker_controller_bp(world):
    return world.get_blueprint_library().find('controller.ai.walker')
